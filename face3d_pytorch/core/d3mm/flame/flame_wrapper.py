"""
https://github.com/soubhiksanyal/FLAME_PyTorch (MIT license)
https://flame.is.tue.mpg.de/index.html
"""

import torch
import torch.nn as nn
from itertools import repeat
from loguru import logger

from .config import get_config
from .flame import FLAME

class FLAMEWrapper(nn.Module):

    def __init__(
        self,
        flame_model_path="face3d_pytorch/assets/flame/FLAME2020/generic_model.pkl",
        static_landmark_embedding_path="face3d_pytorch/assets/flame/flame_static_embedding.pkl",
        dynamic_landmark_embedding_path="face3d_pytorch/assets/flame/flame_dynamic_embedding.npy",
        use_face_contour=True,
        use_3D_translation=True,
        batch_size=8,
        shape_params=100,  # 0-300
        expression_params=50,  # 0-100
        pose_params=6,
    ):
        super().__init__()
        config = get_config()
        config.flame_model_path = flame_model_path
        config.static_landmark_embedding_path = static_landmark_embedding_path
        config.dynamic_landmark_embedding_path = dynamic_landmark_embedding_path
        config.use_face_contour = use_face_contour
        config.use_3D_translation = use_3D_translation
        config.batch_size = batch_size
        config.shape_params = shape_params
        config.expression_params = expression_params
        config.pose_params = pose_params

        self.flamelayer = FLAME(config)
        
        self.batch_size = batch_size
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")

    @torch.no_grad()
    def forward(self, shape_params, expression_params, pose_params, neck_pose=None, eye_pose=None, transl=None):
        # shape_params: (B, shape_params)
        # expression_params: (B, expression_params)
        # pose_params: (B, pose_params)
        #   pose_params[:, :3]: global pose
        #   pose_params[:, 3:]: jaw pose
        # neck_pose: (B, 3)
        #   neck_pose[:, 0]: 绕 x 轴 低头/抬头
        #   neck_pose[:, 1]: 绕 y 轴 左右转头
        #   neck_pose[:, 2]: 绕 z 轴 向肩部侧倾
        # eye_pose: (B, 6)
        #   eye_pose[:, 0:3]  左眼球旋转
        #   eye_pose[:, 3:6]  右眼球旋转
        # transl: (B, 3)，平移量 (tx, ty, tz)
        
        shapes = list(torch.split(shape_params, split_size_or_sections=self.batch_size, dim=0))
        exps = list(torch.split(expression_params, split_size_or_sections=self.batch_size, dim=0))
        poses = list(torch.split(pose_params, split_size_or_sections=self.batch_size, dim=0))
        necks = list(torch.split(neck_pose, split_size_or_sections=self.batch_size, dim=0)) if neck_pose is not None else repeat(None)
        eyes = list(torch.split(eye_pose, split_size_or_sections=self.batch_size, dim=0)) if eye_pose is not None else repeat(None)
        transls = list(torch.split(transl, split_size_or_sections=self.batch_size, dim=0)) if transl is not None else repeat(None)
        # vertice: (B, 5023, 3), landmark: (B, 68, 3)
        verts = []
        landmarks = []
        for shape, exp, pose, neck, eye, trans in zip(shapes, exps, poses, necks, eyes, transls):
            ver, lmk = self.flamelayer(shape, exp, pose, neck, eye, trans)
            verts.append(ver)
            landmarks.append(lmk)
        verts = torch.concat(verts, dim=0)
        landmarks = torch.concat(landmarks, dim=0)

        # 三角面：(9976, 3)，三角面保持不变，所有人脸的三角面一样，3D参数只会影响顶点，不会影响三角面
        faces = self.flamelayer.faces_tensor
        
        # verts: (B, 5023, 3)
        # faces: (9976, 3)
        # landmarks: (B, 68, 3)
        mesh_dict = {
            "verts": verts.detach(),
            "faces": faces.detach(),
            "landmarks": landmarks.detach(),
        }
        return mesh_dict
