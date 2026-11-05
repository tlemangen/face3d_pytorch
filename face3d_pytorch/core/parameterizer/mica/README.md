# mica_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.mica import MICAWrapper

mica = MICAWrapper(
    arcface_checkpoints="face3d_pytorch/assets/mica/arcface.pth",
    mapping_net_checkpoints="face3d_pytorch/assets/mica/mapping_net.pth",
    z_dim=512,
    map_hidden_dim=300,
    n_shape=300,
    mapping_layers=3,
)
mica.eval()
mica.cuda()
```

输入图像，返回 3D 参数字典 和 人脸对齐字典：

```python
params_dict, align_dict = mica(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["shape"]: (B, 300), torch.float32
- align_dict:
    - align_dict["aligned_images"]: (B, 3, 112, 112), torch.float32
