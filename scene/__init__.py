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

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param args: ModelParams containing scene configuration
        :param gaussians: GaussianModel instance
        :param load_iteration: Iteration to load from checkpoint
        :param shuffle: Whether to shuffle camera order
        :param resolution_scales: List of resolution scales to use
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Scene type에 따라 다른 로더 사용
        if args.scene_type == "SuperGlue":
            print("Loading scene with SuperGlue SfM pipeline...")
            scene_info = sceneLoadTypeCallbacks["SuperGlue"](
                args.source_path, 
                args.images, 
                args.eval, 
                args.train_test_exp,
                superglue_config=getattr(args, 'superglue_config', 'outdoor'),
                max_images=getattr(args, 'max_images', 100)
            )
        elif args.scene_type == "SuperGlueCOLMAPHybrid":
            print("Loading scene with SuperGlue + COLMAP hybrid pipeline...")
            scene_info = sceneLoadTypeCallbacks["SuperGlueCOLMAPHybrid"](
                args.source_path, 
                args.images, 
                args.eval, 
                args.train_test_exp,
                superglue_config=getattr(args, 'superglue_config', 'outdoor'),
                max_images=getattr(args, 'max_images', 100)
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Found COLMAP sparse reconstruction")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, 
                args.images, 
                args.depths, 
                args.eval, 
                args.train_test_exp
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, 
                args.white_background, 
                args.depths, 
                args.eval
            )
        else:
            raise ValueError(f"Could not recognize scene type! Please check if {args.source_path} contains valid scene data.")
            
        # 첫 실행시 입력 파일들 복사 및 카메라 정보 저장
        if not self.loaded_iter:
            # PLY 파일 복사
            if os.path.exists(scene_info.ply_path):
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            
            # 카메라 정보를 JSON으로 저장
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        # 카메라 순서 셔플 (일관성 있게)
        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # 장면 범위 설정
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 다양한 해상도 스케일에 대해 카메라 리스트 생성
        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras (scale: {resolution_scale})")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, 
                resolution_scale, 
                args, 
                scene_info.is_nerf_synthetic, 
                False  # is_test_dataset
            )
            
            print(f"Loading Test Cameras (scale: {resolution_scale})")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, 
                resolution_scale, 
                args, 
                scene_info.is_nerf_synthetic, 
                True  # is_test_dataset
            )

        # Gaussian 모델 초기화
        if self.loaded_iter:
            # 체크포인트에서 로드
            ply_path = os.path.join(
                self.model_path,
                "point_cloud",
                f"iteration_{self.loaded_iter}",
                "point_cloud.ply"
            )
            self.gaussians.load_ply(ply_path, args.train_test_exp)
        else:
            # 포인트 클라우드에서 초기화
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, 
                scene_info.train_cameras, 
                self.cameras_extent
            )

    def save(self, iteration):
        """모델 저장"""
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        
        # Gaussian 파라미터 저장
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # Exposure 정보 저장 (있는 경우)
        if hasattr(self.gaussians, 'exposure_mapping') and self.gaussians.exposure_mapping:
            try:
                exposure_dict = {
                    image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                    for image_name in self.gaussians.exposure_mapping
                }
                
                with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                    json.dump(exposure_dict, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save exposure information: {e}")

    def getTrainCameras(self, scale=1.0):
        """학습용 카메라 리스트 반환"""
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """테스트용 카메라 리스트 반환"""
        return self.test_cameras[scale]