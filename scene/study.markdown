
## TODO: 修改sceneLoadTypeCallbacks字典以及3dgs接口

#### NOTE: 场景的初始化
- train.py -> def traning -> Scene() -> ./scene/__init__.py -> class Scene:


```python
    gaussians : GaussianModel  # 加载高斯模型

    sceneLoadTypeCallbacks {} # 含有数据集类型选择的字典，读取后存在scene_info中
```

---
#### NOTE: NeRF场景函数, 关键在scene_info，注意归一化nerf_normalization
- ./scene/dataset.reader.py/sceneLoadTypeCallbacks {} -> def readNerfSyntheticInfo()

```python
    if not os.path.exists(ply_path): # 这里是判断点云文件 "points3d.ply" 是否存在，如果不存在，就生成一个随机的点云数据

    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=train_cam_infos,
                            test_cameras=test_cam_infos,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path,
                            is_nerf_synthetic=True) #包含了点云数据、训练相机列表、测试相机列表、NeRF 归一化信息、点云文件路径以及一个标志位 is_nerf_synthetic

```


---
### NOTE: COLMAP场景函数
- ./scene/dataset.reader.py/sceneLoadTypeCallbacks {} -> def readColmapSceneInfo()
  
```python

    ### KEY FUNCTION
    def readColmapCameras() # 读数据

    def getNerfppNorm()  # 归一化

    def fetchPly()  # 读点云

```
---
### NOTE: 涉及到读取COLMAP数据格式的关键函数
- ./scene/dataset.reader.py/sceneLoadTypeCallbacks {} -> def readColmapCameras()

TODO: 研究输入参数的 数据结构

```python
    # KEY FUNCTION：
    def qvec2rotmat(qvec):
        # 把表示旋转的四元数转换为对应的 3x3 旋转矩阵
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


    def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
    # OUTPUT
    cam_info = CameraInfo(uid=uid,
    R=R,
    T=T,
    FovY=FovY,
    FovX=FovX,
    depth_params=depth_params,
    image_path=image_path,
    image_name=image_name,
    depth_path=depth_path,
    width=width,
    height=height,
    is_test=image_name in test_cam_names_list)

```