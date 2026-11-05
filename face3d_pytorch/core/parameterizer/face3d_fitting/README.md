# face3d_fitting

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.face3d_fitting import Face3DFitting

face3d_fitting = Face3DFitting(bfm_mat_path="face3d_pytorch/assets/face3d/BFM.mat", max_iter=3, input_size=256)
face3d_fitting.eval()
face3d_fitting.cuda()
```

输入图像，返回 3D 参数字典：

```python
params_dict = face3d_fitting(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["id"]: (B, 199), torch.float32
    - params_dict["exp"]: (B, 29), torch.float32
    - params_dict["angle"]: (B, 3), torch.float32
    - params_dict["trans"]: (B, 3), torch.float32
    - params_dict["gamma"]: (B,), torch.float32
