# deca_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.deca import DECAWrapper

deca = DECAWrapper(
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
)
deca.eval()
deca.cuda()
```

输入图像，返回 3D 参数字典 和 人脸对齐字典：

```python
params_dict, align_dict = deca(images)
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
    - params_dict["detail"]: (B, 128), torch.float32
    - params_dict["uv_z"]: (B, 1, 256, 256), torch.float32
- align_dict:
    - align_dict["aligned_images"]: (B, 3, 224, 224), torch.float32
    - align_dict["align_params"]: (B, 3, 3), torch.float32
