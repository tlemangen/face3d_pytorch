# smirk_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.smirk import SMIRKWrapper

smirk = SMIRKWrapper(
    face_landmarker_path="face3d_pytorch/assets/mediapipe/face_landmarker.task",
    smirk_encoder_checkpoints="face3d_pytorch/assets/smirk/smirk_encoder.pth",
    n_exp=50,
    n_shape=300,
)
smirk.eval()
smirk.cuda()
```

输入图像，返回 3D 参数字典 和 人脸对齐字典：

```python
params_dict, align_dict = smirk(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["shape"]: (B, 300), torch.float32
    - params_dict["exp"]: (B, 50), torch.float32
    - params_dict["pose"]: (B, 6), torch.float32
    - params_dict["cam"]: (B, 3), torch.float32
    - params_dict["eyelid"]: (B, 2), torch.float32
- align_dict:
    - align_dict["aligned_images"]: (B, 3, 224, 224), torch.float32
    - align_dict["align_params"]: (B, 3, 3), torch.float32
