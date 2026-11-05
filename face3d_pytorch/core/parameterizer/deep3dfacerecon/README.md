# deep3dfacerecon_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.deep3dfacerecon import Deep3DFaceReconWrapper

deep3dfacerecon = Deep3DFaceReconWrapper(
    similarity_lm3d_all_mat="face3d_pytorch/assets/bfm/similarity_Lm3D_all.mat",
    recon_net_wrapper_checkpoints="face3d_pytorch/assets/deep3dfacerecon/epoch_20.pth",
)
deep3dfacerecon.eval()
deep3dfacerecon.cuda()
```

输入图像，返回 3D 参数字典 和 人脸对齐字典：

```python
params_dict, align_dict = deep3dfacerecon(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["id"]: (B, 80), torch.float32
    - params_dict["exp"]: (B, 64), torch.float32
    - params_dict["tex"]: (B, 80), torch.float32
    - params_dict["angle"]: (B, 3), torch.float32
    - params_dict["gamma"]: (B, 27), torch.float32
    - params_dict["trans"]: (B, 3), torch.float32
- align_dict:
    - align_dict["aligned_images"]: (B, 3, 224, 224), torch.float32
    - align_dict["aligned_landmarks"]: (B, 68, 2), torch.float32
    - align_dict["align_params"]: (B, 5), torch.float32
