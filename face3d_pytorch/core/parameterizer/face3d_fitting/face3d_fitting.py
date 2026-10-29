import torch
import numpy as np
import torch.nn as nn
from tensordict import TensorDict
from insightface.app import FaceAnalysis
from loguru import logger

from ....third_party.face3d.morphable_model import MorphabelModel


class Face3DFitting(nn.Module):
    """https://github.com/yfeng95/face3d"""

    def __init__(self, bfm_mat_path="face3d_pytorch/assets/face3d/BFM.mat", max_iter=3, input_size=256):
        super().__init__()
        self.max_iter = max_iter
        self.input_size = input_size

        self.bfm = MorphabelModel(bfm_mat_path)
        self.kpt_ind = self.bfm.kpt_ind

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "landmark_3d_68"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(self.input_size, self.input_size))
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def fit_from_landmarks(self, landmarks):
        # landmarks: (B, 68, 2), torch.float32

        x = landmarks - self.input_size // 2
        x[..., 1] *= -1

        fitted_shapes, fitted_exps, fitted_scales, fitted_angles, fitted_transs = [], [], [], [], []
        for landmark in x:
            fitted_shape, fitted_exp, fitted_scale, fitted_angle, fitted_trans = self.bfm.fit(
                landmark, self.kpt_ind, max_iter=self.max_iter
            )

            fitted_shapes.append(fitted_shape)
            fitted_exps.append(fitted_exp)
            fitted_scales.append(fitted_scale)
            fitted_angles.append(fitted_angle)
            fitted_transs.append(fitted_trans)
        id = np.stack(fitted_shapes, axis=0).astype(np.float32)  # (B, 199, 1), np.float32
        exp = np.stack(fitted_exps, axis=0).astype(np.float32)  # (B, 29, 1), np.float32
        scale = np.stack(fitted_scales, axis=0).astype(np.float32)  # (B,), np.float32
        angle = np.stack(fitted_angles, axis=0).astype(np.float32)  # (B, 3), np.float32
        trans = np.stack(fitted_transs, axis=0).astype(np.float32)  # (B, 3), np.float32

        return id, exp, scale, angle, trans

    def forward(self, images):
        # images: (B, C, H, W), torch.float32

        images_np = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()  # (B, H, W, C), np.uint8, RGB
        images_np = np.ascontiguousarray(images_np, dtype=np.uint8)
        images_cv = images_np[..., ::-1]  # BGR

        landmarks = []
        for image_cv in images_cv:
            faces = self.app.get(image_cv)
            if len(faces) > 0:
                landmark_3d_68 = faces[0].landmark_3d_68
            landmarks.append(landmark_3d_68[..., :2])
        landmarks = np.stack(landmarks, axis=0).astype(np.float32)

        id, exp, scale, angle, trans = self.fit_from_landmarks(landmarks)

        params_dict = {"id": id[..., 0], "exp": exp[..., 0], "scale": scale, "angle": angle, "trans": trans}

        params_dict = TensorDict(params_dict, batch_size=[images.shape[0]], device=images.device)

        # id: (B, 199), torch.float32
        # exp: (B, 29), torch.float32
        # scale: (B,), torch.float32
        # angle: (B, 3), torch.float32
        # trans: (B, 3), torch.float32
        return params_dict
