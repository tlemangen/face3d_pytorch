import torch
import torch.nn as nn
from loguru import logger

from ...preprocessing.face_alignment import Deep3DFaceReconAlign
from .networks import ReconNetWrapper


class Deep3DFaceReconWrapper(nn.Module):

    def __init__(self, similarity_lm3d_all_mat, recon_net_wrapper_checkpoints):
        super().__init__()
        self.aligner = Deep3DFaceReconAlign(similarity_lm3d_all_mat, mask=None, target_size=224.0, rescale_factor=102.0)

        self.net_recon = ReconNetWrapper(net_recon="resnet50", use_last_fc=False).eval()
        checkpoints = torch.load(recon_net_wrapper_checkpoints)
        self.net_recon.load_state_dict(checkpoints["net_recon"])
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def forward(self, images):
        # images: (B, C, H, W), torch.float32

        align_dict = self.aligner(images)
        aligned_images = align_dict["aligned_images"]

        coeffs = self.net_recon(aligned_images)
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80:144]
        tex_coeffs = coeffs[:, 144:224]
        angles = coeffs[:, 224:227]
        gammas = coeffs[:, 227:254]
        translations = coeffs[:, 254:]
        params_dict = {
            "id": id_coeffs,
            "exp": exp_coeffs,
            "tex": tex_coeffs,
            "angle": angles,
            "gamma": gammas,
            "trans": translations,
        }

        # params_dict:
        #   id: (B, 80)
        #   exp: (B, 64)
        #   tex: (B, 80)
        #   angle: (B, 3)
        #     头部姿态欧拉角，(俯仰, 偏航，滚转)
        #   gamma: (B, 27)
        #     球谐光照参数。可以理解为 gamma.reshape(B, 3, 9)，第二维的 3 表示红、绿、蓝三个颜色通道；第三维的 9 表示每个通道的一组球谐光照基系数；gamma 控制环境光、主要照明方向以及较平滑的高阶光照变化。
        #   trans: (B, 3)
        #     3D平移参数 (tx, ty, tz)
        return params_dict, align_dict
