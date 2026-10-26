import torch
import torch.nn as nn
from tensordict import TensorDict
from loguru import logger

from ...preprocessing.face_alignment import DECAAlign
from .models.encoders import ResnetEncoder
from .models.decoders import Generator


class DECAWrapper(nn.Module):

    def __init__(
        self,
        flame_encoder_checkpoints="face3d_pytorch/assets/deca/flame_encoder.pth",
        detail_encoder_checkpoints="face3d_pytorch/assets/deca/detail_encoder.pth",
        detail_decoder_checkpoints="face3d_pytorch/assets/deca/detail_decoder.pth",
        n_shape=100,
        n_tex=50,
        n_exp=50,
        n_pose=6,
        n_cam=3,
        n_light=27,
        n_detail=128,
        max_z=0.01,
    ):
        super().__init__()
        self.aligner = DECAAlign()

        n_param = n_shape + n_tex + n_exp + n_pose + n_cam + n_light
        n_cond = n_exp + 3  # exp + jaw pose
        self.E_flame = ResnetEncoder(outsize=n_param).eval()
        self.E_detail = ResnetEncoder(outsize=n_detail).eval()
        self.D_detail = Generator(latent_dim=n_detail + n_cond, out_channels=1, out_scale=max_z, sample_mode='bilinear').eval()
        self.E_flame.load_state_dict(torch.load(flame_encoder_checkpoints))
        self.E_detail.load_state_dict(torch.load(detail_encoder_checkpoints))
        self.D_detail.load_state_dict(torch.load(detail_decoder_checkpoints))

        self.param_dict = {"shape": n_shape, "tex": n_tex, "exp": n_exp, "pose": n_pose, "cam": n_cam, "light": n_light}

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def _decompose_code(self, code):
        '''Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in self.param_dict:
            end = start + int(self.param_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    @torch.no_grad()
    def forward(self, images):
        # images: (B, C, H, W), torch.float32, RGB

        align_dict = self.aligner(images)
        aligned_images = align_dict["aligned_images"]

        parameters = self.E_flame(aligned_images)
        params_dict = self._decompose_code(parameters)

        detail = self.E_detail(aligned_images)
        params_dict["detail"] = detail
        
        jaw_pose = params_dict["pose"][:, 3:]
        exp = params_dict["exp"]
        detail_condition = torch.cat([jaw_pose, exp, detail], dim=-1)
        uv_z = self.D_detail(detail_condition)  # (B, 1, uv_size, uv_size)
        params_dict["uv_z"] = uv_z
            

        # params_dict:
        #   shape: (T, 100)
        #   tex: (T, 50)
        #   exp: (T, 50)
        #   pose: (T, 6)
        #   cam: (T, 3)
        #   light: (T, 9, 3)
        #   detail: (B, 128)
        #   uv_z: (B, 1, 256, 256)
        return params_dict, align_dict
