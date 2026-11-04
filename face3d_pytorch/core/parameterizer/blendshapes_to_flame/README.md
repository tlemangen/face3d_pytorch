# blendshapes2flame_wrapper

# Useage

实例化：

```python
from face3d_pytorch.core.parameterizer.blendshapes_to_flame import Blendshapes2FLAMEWrapper

b2f = Blendshapes2FLAMEWrapper(
    face_landmarker_path="face3d_pytorch/assets/mediapipe/face_landmarker.task",
    mappings_path="face3d_pytorch/assets/blendshapes_to_flame/mappings",
)
b2f.eval()
b2f.cuda()
```

输入图像，返回 3D 参数字典 和 mediapipe 中间量字典：

```python
params_dict, mediapipe_dict = b2f(images)
```

args:

- images: (B, C, H, W), torch.float32

return:

- params_dict:
    - params_dict["exp"]: (B, 100), torch.float32
    - params_dict["pose"]: (B, 6), torch.float32
    - params_dict["eye_pose"]: (B, 6), torch.float32
    - params_dict["trans"]: (B, 3), torch.float32
- mediapipe_dict:
    - mediapipe_dict["landmarks_3d"]: (B, 478, 3), torch.float32
    - mediapipe_dict["landmarks_68"]: (B, 68, 3), torch.float32
    - mediapipe_dict["blendshapes"]: (B, 52), torch.float32
    - mediapipe_dict["facial_transformation_matrixes"]: (B, 4, 4), torch.float32

## 方法

> face3d_pytorch.core.parameterizer.blendshapes_to_flame.Blendshapes2FLAMEWrapper.images2blendshapes(images) -> mediapipe_dict

输入图像，返回 mediapipe 字典。

args:

- images: (B, C, H, W), torch.float32

return:

- mediapipe_dict:
    - mediapipe_dict["landmarks_3d"]: (B, 478, 3), torch.float32
    - mediapipe_dict["landmarks_68"]: (B, 68, 3), torch.float32
    - mediapipe_dict["blendshapes"]: (B, 52), torch.float32
    - mediapipe_dict["facial_transformation_matrixes"]: (B, 4, 4), torch.float32

> face3d_pytorch.core.parameterizer.blendshapes_to_flame.Blendshapes2FLAMEWrapper.blendshapes2params(blendshapes, facial_transformation_matrixes) -> params_dict

输入 blendshapes 和 facial_transformation_matrixes，返回 3D 参数字典。

args:

- blendshapes: (B, 52), torch.float32
- facial_transformation_matrixes: (B, 4, 4), torch.float32

return:

- params_dict:
    - params_dict["exp"]: (B, 100), torch.float32
    - params_dict["pose"]: (B, 6), torch.float32
    - params_dict["eye_pose"]: (B, 6), torch.float32
    - params_dict["trans"]: (B, 3), torch.float32

useage:

```python
params_dict = Blendshapes2FLAMEWrapper.blendshapes2params(mediapipe_dict["blendshapes"], mediapipe_dict["facial_transformation_matrixes"])
```