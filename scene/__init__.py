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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel  # 加载高斯模型

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:  # 看上去是想加载训练迭代数据，如果没有指定迭代数，就搜索最大的迭代数
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))  # 搜索最大的迭代数
            else:
                self.loaded_iter = load_iteration  # 如果指定了迭代数，就加载指定的迭代数据
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}  # 初始化训练和测试相机字典
        self.test_cameras = {} 

        if os.path.exists(os.path.join(args.source_path, "sparse")):  # 这里是判断输入路径下是否存在 "sparse" 文件夹，如果存在，说明这是一个 Colmap 场景
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
            
        
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):  # 这里是判断输入路径下是否存在 "transforms_train.json" 文件，如果存在，说明这是一个 Blender 数据集
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
                   
        else: #  
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:  # 如果没有加载迭代数据，就将场景信息中的点云数据保存到模型路径下的 "input.ply" 文件中，并将相机信息保存到 "cameras.json" 文件中
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())  # 将场景信息中的点云数据复制到模型路径下的 "input.ply" 文件中
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)  # 将测试相机列表中的相机添加到 camlist 中
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)  # 将训练相机列表中的相机添加到 camlist 中
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))  # 将 camlist 中的每个相机转换为 JSON 格式，并添加到 json_cams 列表中
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)  # 将 json_cams 列表写入到 "cameras.json" 文件中，保存相机信息

        if shuffle:  # 如果需要打乱训练和测试相机列表，就使用 random.shuffle 函数对它们进行随机打乱，以确保多分辨率训练的一致性
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]  # 这里是将场景信息中的 NeRF 归一化半径保存到 cameras_extent 变量中，这个变量可能用于后续的相机处理或训练过程中

        for resolution_scale in resolution_scales:  # 对于每个分辨率缩放比例，使用 cameraList_from_camInfos 函数从场景信息中的训练和测试相机列表中创建相应的相机列表，并将它们保存到 train_cameras 和 test_cameras 字典中，以便后续使用
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:  # 如果已经加载了迭代数据，就使用高斯模型的 load_ply 方法从模型路径下的 "point_cloud/iteration_{loaded_iter}/point_cloud.ply" 文件中加载点云数据，并将训练测试分割信息传递给该方法，以便正确处理点云数据
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:  # 如果没有加载迭代数据，就使用高斯模型的 create_from_pcd 方法从场景信息中的点云数据创建高斯模型，并将训练相机列表和相机范围信息传递给该方法，以便正确处理点云数据并创建高斯模型
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):  #用于保存当前的高斯模型状态和相关信息到指定的路径下，以便后续使用或恢复训练。该方法接受一个 iteration 参数，表示当前的训练迭代数。
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()# 
            for image_name in self.gaussians.exposure_mapping
        } # 将曝光信息保存到字典中，键是图像名称，值是从高斯模型中获取的曝光信息，并将其转换为列表格式

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:  # 将当前的曝光信息保存到 "exposure.json" 文件中，保存的内容是一个字典，其中键是图像名称，值是从高斯模型中获取的曝光信息，并将其转换为列表格式以便 JSON 序列化。
            json.dump(exposure_dict, f, indent=2) # 将曝光信息保存到 "exposure.json" 文件中，以便后续使用

    def getTrainCameras(self, scale=1.0):   # 用于获取训练相机列表的方法。该方法接受一个可选的 scale 参数，表示分辨率缩放比例，默认为 1.0。根据传入的 scale 参数，从 train_cameras 字典中获取对应分辨率缩放比例的训练相机列表，并返回该列表。
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
