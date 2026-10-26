"""
https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from loguru import logger

from .mp_2_flame import MP_2_FLAME
from .landmark_mapping import mediapipe_to_ibug68


class Blendshapes2FLAMEWrapper(nn.Module):

    def __init__(
        self,
        face_landmarker_path="face3d_pytorch/assets/mediapipe/face_landmarker.task",
        mappings_path="face3d_pytorch/assets/blendshapes_to_flame/mappings",
    ):
        super().__init__()
        # download model weights from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
        base_options = mp.tasks.BaseOptions(model_asset_path=face_landmarker_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.mp2flame = MP_2_FLAME(mappings_path=mappings_path)

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def preprocessing(self, images):
        # images: (B, C, H, W)

        # MediaPipe 的 SRGB 输入需要 RGB；这里只改变布局和数值范围，不交换通道。
        images_cv = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()
        images_cv = np.ascontiguousarray(images_cv, dtype=np.uint8)

        return images_cv

    @staticmethod
    def extract_pose_from_matrix(transformation_matrix):
        matrix = np.asarray(transformation_matrix, dtype=np.float64).reshape(4, 4)

        translation = matrix[:3, 3].copy()

        # MediaPipe 的左上 3x3 理论上是旋转矩阵；
        # 使用 SVD 去掉可能存在的尺度和数值误差。
        A = matrix[:3, :3]
        U, _, Vt = np.linalg.svd(A)
        rotation_matrix = U @ Vt

        # 避免得到反射矩阵
        if np.linalg.det(rotation_matrix) < 0:
            U[:, -1] *= -1
            rotation_matrix = U @ Vt

        # 直接转换为 FLAME 所需的轴角，单位为弧度
        rotation_vec, _ = cv2.Rodrigues(rotation_matrix)
        rotation_vec = rotation_vec.reshape(3)

        return {"translation": translation, "rotation": rotation_vec}

    def forward(self, images):
        # images: (B, C, H, W)

        images_cv = self.preprocessing(images)

        lmks_3d_batch = []
        blendshapes_batch = []
        exp_batch = []
        jaw_pose_batch = []
        eye_pose_batch = []
        rotation_batch = []
        translation_batch = []
        for image_cv in images_cv:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_cv)
            height, width = image.height, image.width
            detection_result = self.landmarker.detect(image)

            lmks_3d = detection_result.face_landmarks[0]
            lmks_3d = np.array(list(map(lambda l: np.array([l.x, l.y, l.z]), lmks_3d)))  # [478, 3]
            # MediaPipe 的 x、y 分别按图像宽度和高度归一化。
            # z 与 x 使用相同的归一化尺度，因此也乘 width，避免三轴单位混用。
            lmks_3d[:, 0] = lmks_3d[:, 0] * width
            lmks_3d[:, 1] = lmks_3d[:, 1] * height
            lmks_3d[:, 2] = lmks_3d[:, 2]

            blendshapes = detection_result.face_blendshapes[0]
            blendshapes = np.array([list(map(lambda b: b.score, blendshapes))])  # [1, 52]

            exp, jaw_pose, eye_pose = self.mp2flame.convert(blendshape_scores=blendshapes)  # (1, 100), (1, 3), (1, 6)

            # rotation: 旋转向量，euler角 (pitch, yaw, roll)
            # translation: 平移向量 (tx, ty, tz)
            # rotation_vec, translation_vec = compute_head_pose_from_mp_landmarks_3d(
            #     face_landmarks=lmks_3d, img_h=height, img_w=width
            # )
            # rotation = rotation_vec.T  # (1, 3)
            # translation = translation_vec.T  # (1, 3)

            pose = self.extract_pose_from_matrix(detection_result.facial_transformation_matrixes[0])
            rotation = pose["rotation"][None, :]
            translation = pose["translation"][None, :]

            lmks_3d_batch.append(torch.from_numpy(lmks_3d))
            blendshapes_batch.append(torch.from_numpy(blendshapes))
            exp_batch.append(torch.from_numpy(exp))
            jaw_pose_batch.append(torch.from_numpy(jaw_pose))
            eye_pose_batch.append(torch.from_numpy(eye_pose))
            rotation_batch.append(torch.from_numpy(rotation))
            translation_batch.append(torch.from_numpy(translation))
        lmks_3d_batch = torch.stack(lmks_3d_batch, dim=0).to(images.device, dtype=torch.float32)
        blendshapes_batch = torch.concat(blendshapes_batch, dim=0).to(images.device, dtype=torch.float32)
        exp_batch = torch.concat(exp_batch, dim=0).to(images.device, dtype=torch.float32)
        jaw_pose_batch = torch.concat(jaw_pose_batch, dim=0).to(images.device, dtype=torch.float32)
        eye_pose_batch = torch.concat(eye_pose_batch, dim=0).to(images.device, dtype=torch.float32)
        rotation_batch = torch.concat(rotation_batch, dim=0).to(images.device, dtype=torch.float32)
        translation_batch = torch.concat(translation_batch, dim=0).to(images.device, dtype=torch.float32)

        exp = exp_batch.detach()
        pose = torch.concat([rotation_batch, jaw_pose_batch], dim=-1).detach()
        eye_pose = eye_pose_batch.detach()
        trans = translation_batch.detach()
        lmks_68_batch = mediapipe_to_ibug68(lmks_3d_batch).detach()

        # trans 属于 MediaPipe canonical face 坐标系，仅保留作诊断信息；
        # 它不能直接作为 FLAME transl 或 render_shape 相机使用。
        params_dict = {"exp": exp, "pose": pose, "eye_pose": eye_pose, "trans": trans}
        mediapipe_dict = {
            "landmarks_3d": lmks_3d_batch.detach(),
            "landmarks_68": lmks_68_batch,
            "blendshapes": blendshapes_batch.detach(),
        }

        return params_dict, mediapipe_dict
