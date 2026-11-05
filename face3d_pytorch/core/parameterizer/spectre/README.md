# spectre_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.spectre import SPECTREWrapper

spectre = SPECTREWrapper(
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
)
spectre.eval()
spectre.cuda()
```

输入图像，返回 3D 参数字典 和 人脸对齐字典：

```python
params_dict, align_dict = spectre(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["shape"]: (B, 100), torch.float32
    - params_dict["tex"]: (B, 50), torch.float32
    - params_dict["exp"]: (B, 50), torch.float32
    - params_dict["pose"]: (B, 6), torch.float32
    - params_dict["cam"]: (B, 3), torch.float32
    - params_dict["light"]: (B, 9, 3), torch.float32
- align_dict:
    - align_dict["aligned_images"]: (B, 3, 224, 224), torch.float32
    - align_dict["align_params"]: (B, 3, 3), torch.float32
