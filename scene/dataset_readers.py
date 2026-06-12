#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple): # 
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    """
    计算 NeRF++ 归一化参数，用于场景的标准化处理。

    该函数通过遍历相机信息列表，提取所有相机的中心点坐标，
    计算这些中心点的几何中心以及最大覆盖半径（并增加 10% 的余量）。
    返回的平移量和半径可用于后续将场景坐标归一化到单位球或特定范围内，
    以便在创建高斯模型或进行渲染时保持数值稳定性。

    Parameters:
        cam_info (list): 相机信息对象列表。每个对象应包含属性 'R' (旋转矩阵) 和 'T' (平移向量)。

    Returns:
        dict: 包含归一化参数的字典：
            - "translate" (np.ndarray): 平移向量，值为负的几何中心坐标，用于将场景中心移至原点。
            - "radius" (float): 归一化半径，为最大相机中心距离的 1.1 倍。
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True) # 均值
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    # 提取所有相机的世界坐标系中心点
    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)  # 世界坐标系到相机坐标系
        C2W = np.linalg.inv(W2C)  # 相机坐标系到世界坐标系
        cam_centers.append(C2W[:3, 3:4])  # 世界坐标系中心

    # 计算相机中心点的几何中心及最大对角距离
    center, diagonal = get_center_and_diag(cam_centers)  #
    
    # 增加 10% 的半径余量以确保包含所有相机视角
    radius = diagonal * 1.1

    translate = -center  # 计算平移向量

    return {"translate": translate, "radius": radius}  # 返回平移向量和半径

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):  # 
        sys.stdout.write('\r')  # 这里是使用 sys.stdout.write 函数在命令行中输出当前正在读取的相机的索引和总相机数量，以便用户能够实时了解读取进度。这个输出会覆盖之前的输出，使得用户能够看到最新的读取状态，而不会产生多行输出。
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]  # 提取外参
        intr = cam_intrinsics[extr.camera_id]  # 提取内参
        height = intr.height  # 获取图像的高度
        width = intr.width  # 获取图像的宽度

        uid = intr.id
        # NOTE: Why this matrix need transpose? Because COLMAP stores the rotation in a transposed way, and the CUDA code expects it to be transposed back to match the original rotation. This is likely due to the way COLMAP represents rotations and how the CUDA code processes them, ensuring that the rotation information is correctly interpreted during rendering and training.
        R = np.transpose(qvec2rotmat(extr.qvec)) # 这里是将相机的旋转向量（quaternion）转换为旋转矩阵，并对其进行转置操作，以便在后续的相机处理和训练过程中能够正确地使用这个旋转矩阵来表示相机的旋转信息。
        T = np.array(extr.tvec)  # 这里是将相机的平移向量转换为 NumPy 数组，以便在后续的相机处理和训练过程中能够正确地使用这个平移向量来表示相机的位置和移动信息。

        if intr.model=="SIMPLE_PINHOLE":  # 方形像素
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)  # 这里是根据相机的焦距和图像的高度来计算相机的垂直视场角（FovY）。这个计算可能是为了在后续的相机处理和训练过程中能够正确地使用这个视场角来表示相机的视野范围。
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":  # 非方形像素
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1  # 这里是计算相机名称中需要去除的后缀长度，以便在后续的深度参数文件路径构建和图像名称提取过程中能够正确地处理相机名称。这个计算可能是为了适应不同格式的相机名称，例如 "image.png" 中的 ".png" 后缀，确保在构建深度参数文件路径和提取图像名称时能够正确地去除这些后缀部分。
        depth_params = None
        if depths_params is not None:  # 
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    """
    将点云数据保存为 PLY 格式文件。

    该函数接收点的坐标和颜色信息，自动生成零法向量，
    并将其组合写入指定的 PLY 文件中。

    Parameters:
        path (str): 输出 PLY 文件的路径。
        xyz (numpy.ndarray): 点云的坐标数组，形状为 (N, 3)，数据类型通常为 float32 或 float64。
        rgb (numpy.ndarray): 点云的颜色数组，形状为 (N, 3)，数据类型通常为 uint8。

    Returns:
        None
    """
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    
    # 这里是尝试读取 Colmap 场景的相机外参和内参文件，如果读取失败，则尝试读取文本格式的相机外参和内参文件。这个过程是为了兼容不同格式的 Colmap 输出文件，以确保能够正确加载场景信息。
    try:  
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
    # 这里是构建深度参数文件的路径，假设深度参数文件位于 Colmap 场景的 "sparse/0" 文件夹下，并且命名为 "depth_params.json"。这个文件可能包含与深度图相关的参数信息，例如深度图的缩放比例、偏移量等，这些信息可能在后续的场景加载和处理过程中使用到。
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json") 
    
    
    ## if depth_params_file isnt there AND depths file is here -> throw error
    # 这里是检查深度参数文件是否存在，如果存在，则尝试打开该文件并读取其中的深度参数信息。如果文件不存在，则打印错误信息并退出程序。
    depths_params = None
    # 这里是检查是否指定了深度图文件夹，如果指定了，则尝试打开深度参数文件并读取其中的深度参数信息。如果没有指定深度图文件夹，则不需要读取深度参数文件，因此可以跳过这个步骤。
    if depths != "":  
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
            
    # 这里是根据输入参数 eval 的值来确定测试相机列表。如果 eval 参数为 True，则从 Colmap 场景的 "sparse/0/test.txt" 文件中读取测试相机的名称列表，并将其保存在 test_cam_names_list 变量中。这个文件可能包含了需要用于评估的测试相机的名称，以便在后续的场景加载和处理过程中正确地标识哪些相机是测试相机。
    if eval: 
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) # 这里是对读取到的相机信息列表进行排序，排序的依据是相机的图像名称。这个排序操作可能是为了确保相机信息列表中的相机按照图像名称的顺序排列，以便在后续的场景加载和处理过程中能够正确地对应相机信息和图像数据。
    
    # 这里是根据输入参数 train_test_exp 的值来确定训练相机列表。如果 train_test_exp 参数为 True，则将所有相机信息都包含在训练相机列表中；如果 train_test_exp 参数为 False，则只将非测试相机的信息包含在训练相机列表中。这个操作可能是为了根据不同的训练和评估需求来灵活地选择哪些相机信息应该用于训练，以便在后续的场景加载和处理过程中能够正确地标识哪些相机是训练相机。
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]  
    
    # 这里是根据相机信息中的 is_test 字段来确定测试相机列表。这个操作可能是为了将相机信息列表中的测试相机的信息提取出来，并保存在 test_cam_infos 变量中，以便在后续的场景加载和处理过程中能够正确地标识哪些相机是测试相机。
    test_cam_infos = [c for c in cam_infos if c.is_test]  
    
    # 这里是调用 getNerfppNorm 函数来计算 NeRF 归一化信息，并将其保存在 nerf_normalization 变量中。这个函数可能会根据训练相机的信息来计算出一个中心点和一个半径，用于在 NeRF 模型中进行归一化处理，以便在后续的场景加载和处理过程中能够正确地处理相机信息并创建高斯模型。0
    nerf_normalization = getNerfppNorm(train_cam_infos)  
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        # 这里是使用 fetchPly 函数从 "points3D.ply" 文件中加载点云数据，并将其保存在 pcd 变量中，以便后续的高斯模型创建和训练过程能够使用这个点云数据。
        pcd = fetchPly(ply_path)  
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    """
    从 transforms JSON 文件中读取相机信息和图像数据。

    该函数解析 NeRF 风格的 transforms 文件，提取相机的内外参、视野角度（FOV），
    并处理图像数据（包括去背景和格式转换）。同时可选地关联深度图路径。

    参数:
        path (str): 数据集根目录路径。
        transformsfile (str): transforms JSON 文件的文件名。
        depths_folder (str): 深度图所在的文件夹路径。如果为空字符串，则不加载深度图。
        white_background (bool): 是否使用白色背景。如果为 True，透明区域填充白色；否则填充黑色。
        is_test (bool): 标记当前加载的数据集是否用于测试阶段。
        extension (str, optional): 图像文件的扩展名，默认为 ".png"。

    返回:
        list[CameraInfo]: 包含每个视角相机信息的 CameraInfo 对象列表。
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)#  
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):  # 这里是判断点云文件 "points3d.ply" 是否存在，如果不存在，就生成一个随机的点云数据，并将其保存到 "points3d.ply" 文件中。这个点云数据是为了在没有 Colmap 数据的情况下提供一个初始的点云，以便后续的高斯模型创建和训练过程能够正常进行。
        # Since this data set has no colmap data, we start with random points 
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))  # 这里是创建一个 BasicPointCloud 对象，包含随机生成的点云数据。点云数据由随机生成的 xyz 坐标和对应的颜色值组成，其中颜色值是通过 SH2RGB 函数将随机生成的 shs 值转换为 RGB 颜色值，并且将法线设置为全零。这个点云数据将被保存到 "points3d.ply" 文件中，以便后续的高斯模型创建和训练过程能够使用这个初始的点云数据。

        storePly(ply_path, xyz, SH2RGB(shs) * 255)  # 这里是调用 storePly 函数，将随机生成的点云数据保存到 "points3d.ply" 文件中。函数接受点云的 xyz 坐标和对应的 RGB 颜色值作为输入，并将它们写入到指定的 PLY 文件中，以便后续的高斯模型创建和训练过程能够使用这个初始的点云数据。
    try:
        pcd = fetchPly(ply_path) # 这里是使用 fetchPly 函数从 "points3d.ply" 文件中加载点云数据，并将其保存到 pcd 变量中，以便后续的高斯模型创建和训练过程能够使用这个点云数据。
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True) # 这里是创建一个 SceneInfo 对象，包含了点云数据、训练相机列表、测试相机列表、NeRF 归一化信息、点云文件路径以及一个标志位 is_nerf_synthetic，表示这个场景是否是 NeRF 合成数据集。这个 SceneInfo 对象将被返回给调用者，以便后续的高斯模型创建和训练过程能够使用这些场景信息。
    return scene_info

# 直接读取深度相机的场景信息
def readRealSceneInfo(path, images, depths):
    
    
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=Flase)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Realdata": readRealSceneInfo,
}  # 这里创建了一个字典，将数据集类型（Colmap、Blender、Realdata）映射到对应的数据读取函数。