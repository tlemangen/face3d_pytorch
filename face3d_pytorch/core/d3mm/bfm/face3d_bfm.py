import torch
import numpy as np
import torch.nn as nn
from loguru import logger

from ....third_party.face3d.morphable_model import MorphabelModel


class Face3DBFM(nn.Module):
    """https://github.com/yfeng95/face3d"""

    def __init__(self, bfm_mat_path="face3d_pytorch/assets/face3d/BFM.mat", img_size=256, device="cuda"):
        super().__init__()
        self.img_size = img_size
        self.device = device

        self.bfm = MorphabelModel(bfm_mat_path)
        self.faces = torch.from_numpy(self.bfm.triangles).to(device)
        self.kpt_ind = self.bfm.kpt_ind
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def forward(self, id_params, exp_params, scale_params, angle_params, trans_params):
        # id_params: (B, 199), torch.float32
        # exp_params: (B, 29), torch.float32
        # scale_params: (B,), torch.float32
        # angle_params: (B, 3), torch.float32
        # trans_params: (B, 3), torch.float32

        id_params_np = id_params[..., None].detach().cpu().numpy()  # (B, 199, 1), np.float32
        exp_params_np = exp_params[..., None].detach().cpu().numpy()  # (B, 29, 1), np.float32
        scale_params_np = scale_params.detach().cpu().numpy()  # (B,), np.float32
        angle_params_np = angle_params.detach().cpu().numpy()  # (B, 3), np.float32
        trans_params_np = trans_params.detach().cpu().numpy()  # (B, 3), np.float32

        transformed_vertices_list = []
        landmarks_list = []
        for id, exp, scale, angle, trans in zip(id_params_np, exp_params_np, scale_params_np, angle_params_np, trans_params_np):
            fitted_vertices = self.bfm.generate_vertices(id, exp)
            transformed_vertices = self.bfm.transform(fitted_vertices, scale, angle, trans)
            landmarks = transformed_vertices[self.kpt_ind, :2]

            transformed_vertices_list.append(transformed_vertices)
            landmarks_list.append(landmarks)
        verts_np = np.stack(transformed_vertices_list, axis=0)
        landmarks_np = np.stack(landmarks_list, axis=0)

        verts = torch.from_numpy(verts_np).to(dtype=torch.float32, device=self.device)
        landmarks = torch.from_numpy(landmarks_np).to(dtype=torch.float32, device=self.device)

        landmarks[..., 1] *= -1
        landmarks += self.img_size // 2

        mesh_dict = {"verts": verts, "faces": self.faces, "landmarks": landmarks}
        
        # verts: (B, 53215, 3), torch.float32
        # faces: (105840, 3), torch.float32
        # landmarks: (B, 68, 2), torch.float32
        return mesh_dict
