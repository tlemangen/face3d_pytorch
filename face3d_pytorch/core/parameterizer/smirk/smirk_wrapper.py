import torch
import torch.nn as nn
from loguru import logger

from ...preprocessing.face_alignment import SMIRKAlign
from .smirk_encoder import SmirkEncoder


class SMIRKWrapper(nn.Module):

    def __init__(
        self,
        face_landmarker_path="face3d_pytorch/assets/mediapipe/face_landmarker.task",
        smirk_encoder_checkpoints="face3d_pytorch/assets/smirk/smirk_encoder.pth",
        n_exp=50,
        n_shape=300,
    ):
        super().__init__()
        self.aligner = SMIRKAlign(face_landmarker_path).eval()

        self.smirk_encoder = SmirkEncoder(n_exp=n_exp, n_shape=n_shape).eval()
        self.smirk_encoder.load_state_dict(torch.load(smirk_encoder_checkpoints))

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def forward(self, images):
        # images: (B, C, H, W), torch.float32

        align_params = self.aligner(images)

        codes = self.smirk_encoder(align_params["aligned_images"])

        # pose_params torch.Size([58, 3])
        # cam torch.Size([58, 3])
        # shape_params torch.Size([58, 300])
        # expression_params torch.Size([58, 50])
        # eyelid_params torch.Size([58, 2])
        # jaw_params torch.Size([58, 3])
        shape = codes["shape_params"]  # (B, 300)
        exp = codes["expression_params"]  # (B, 50)
        head_pose = codes["pose_params"]  # (B, 3)
        jaw_pose = codes["jaw_params"]  # (B, 3)
        pose = torch.concat([head_pose, jaw_pose], dim=-1)  # (B, 6)
        cam = codes["cam"]  # (B, 3)
        eyelid = codes["eyelid_params"]  # (B, 2), 全0

        params_dict = {"shape": shape, "exp": exp, "pose": pose, "cam": cam, "eyelid": eyelid}
        return params_dict, align_params
