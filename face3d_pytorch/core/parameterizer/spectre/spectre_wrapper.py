import torch
import torch.nn as nn
from tensordict import TensorDict
from loguru import logger

from ...preprocessing.face_alignment import SPECTREAlign
from .models.encoders import ResnetEncoder, PerceptualEncoder


class SPECTREWrapper(nn.Module):

    def __init__(
        self,
        retinaface_resnet50_final_pth="face3d_pytorch/assets/ibug/face_detection/retina_face/weights/Resnet50_Final.pth",
        fan_2dfan2_pth="face3d_pytorch/assets/ibug/face_alignment/fan/weights/2dfan2.pth",
        flame_encoder_checkpoints="face3d_pytorch/assets/spectre/flame_encoder.pth",
        exp_encoder_checkpoints="face3d_pytorch/assets/spectre/exp_encoder.pth",
        n_shape=100,
        n_tex=50,
        n_exp=50,
        n_pose=6,
        n_cam=3,
        n_light=27,
        device="cuda",
    ):
        super().__init__()
        self.aligner = SPECTREAlign(retinaface_resnet50_final_pth, fan_2dfan2_pth, device)

        n_param = n_shape + n_tex + n_exp + n_pose + n_cam + n_light
        self.E_flame = ResnetEncoder(outsize=n_param).eval()
        self.E_expression = PerceptualEncoder("mobilenetv2").eval()
        self.E_flame.load_state_dict(torch.load(flame_encoder_checkpoints))
        self.E_expression.load_state_dict(torch.load(exp_encoder_checkpoints))

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
    def forward(self, video):
        # video: (T, C, H, W), torch.float32, RGB

        align_dict = self.aligner(video)
        aligned_frames_list = align_dict["aligned_images"]
        tforms_list = align_dict["align_params"]

        params_dict_list = []
        align_dict_list = []
        for idx, (aligned_frames, tforms) in enumerate(zip(aligned_frames_list, tforms_list)):
            parameters = self.E_flame(aligned_frames)
            params_dict = self._decompose_code(parameters)

            deca_exp = params_dict["exp"].clone()
            deca_jaw_pose = params_dict["pose"][..., 3:].clone()

            delta_exp, delta_jaw_pose = self.E_expression(aligned_frames)
            params_dict["exp"] = deca_exp + delta_exp
            params_dict["pose"][..., 3:] = deca_jaw_pose + delta_jaw_pose

            params_dict = TensorDict(params_dict, batch_size=parameters.shape[0])
            align_dict = {"aligned_images": aligned_frames, "align_params": tforms}
            align_dict = TensorDict(align_dict, batch_size=parameters.shape[0])
            if idx == 0 and idx == len(aligned_frames_list) - 1:
                pass
            elif idx == 0:
                params_dict = params_dict[:-2]
                align_dict = align_dict[:-2]
            elif idx == len(aligned_frames_list) - 1:
                params_dict = params_dict[2:]
                align_dict = align_dict[2:]
            else:
                params_dict = params_dict[2:-2]
                align_dict = align_dict[2:-2]
            params_dict_list.append(params_dict)
            align_dict_list.append(align_dict)
        params_dict = torch.cat(params_dict_list, dim=0)
        align_dict = torch.cat(align_dict_list, dim=0)
        params_dict = params_dict[2:-2]
        align_dict = align_dict[2:-2]

        # params_dict:
        #   shape: (T, 100)
        #   tex: (T, 50)
        #   exp: (T, 50)
        #   pose: (T, 6)
        #   cam: (T, 3)
        #   light: (T, 9, 3)
        return params_dict, align_dict
