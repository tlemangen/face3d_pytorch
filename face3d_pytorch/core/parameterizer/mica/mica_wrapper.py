import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ...preprocessing.face_alignment import MICAAlign
from .models.arcface import Arcface
from .models.generator import MappingNetwork


class MICAWrapper(nn.Module):

    def __init__(
        self,
        arcface_checkpoints="face3d_pytorch/assets/mica/arcface.pth",
        mapping_net_checkpoints="face3d_pytorch/assets/mica/mapping_net.pth",
        z_dim=512,
        map_hidden_dim=300,
        n_shape=300,
        mapping_layers=3,
    ):
        super().__init__()
        self.aligner = MICAAlign()
        
        self.arcface = Arcface(pretrained_path=None).eval()

        # checkpoint = torch.load("face3d_pytorch/assets/mica/mica.tar")
        # self.arcface.load_state_dict(checkpoint["arcface"])
        self.arcface.load_state_dict(torch.load(arcface_checkpoints))

        self.regressor = MappingNetwork(z_dim, map_hidden_dim, n_shape, mapping_layers).eval()
        # self.regressor.load_state_dict(OrderedDict((k.replace("regressor.", ""), v) for k, v in checkpoint["flameModel"].items() if "regressor." in k))
        self.regressor.load_state_dict(torch.load(mapping_net_checkpoints))
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")
    
    @torch.no_grad()
    def forward(self, images):
        # images: (B, C, H, W), torch.float32, RGB
        
        align_dict = self.aligner(images)  # (B, C, 112, 112), torch.float32, normalized
        aligned_images = align_dict["aligned_images"]
        
        id_code = self.arcface(aligned_images)
        id_code = F.normalize(id_code)
        shape = self.regressor(id_code)
        
        params_dict = {
            "shape": shape.detach()
        }
        return params_dict, align_dict
