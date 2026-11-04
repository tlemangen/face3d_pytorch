# flame_wrapper

## Useage

实例化：

```python
from face3d_pytorch.core.d3mm.flame import FLAMEWrapper

flame = FLAMEWrapper(
    flame_model_path="face3d_pytorch/assets/flame/FLAME2020/generic_model.pkl",
    static_landmark_embedding_path="face3d_pytorch/assets/flame/flame_static_embedding.pkl",
    dynamic_landmark_embedding_path="face3d_pytorch/assets/flame/flame_dynamic_embedding.npy",
    n_shape=100,
    n_exp=50,
    n_pose=6,
)
flame.cuda()
```

输入形状、表情和姿态等参数，返回一个包括顶点和三角面等信息的字典：

```python
mesh_dict = flame(shape_params, expression_params, pose_params, neck_pose, eye_pose, splite_size=16)
```

args:

- shape_params: (B, `n_shape`), torch.float32. `n_shape` 范围是 0-300.
- expression_params: (B, `n_exp`), torch.float32. `n_exp` 范围是 0-100.
- pose_params: (B, `n_pose`), torch.float32.
- neck_pose: (B, 3), torch.float32.
- eye_pose: (B, 6), torch.float32.
- splite_size: 批次大小.

return:

- mesh_dict["verts"]: (T, 5023, 3), torch.float32. 顶点.
- mesh_dict["faces"]: (F, 3), torch.int32. 三角面.
- mesh_dict["landmarks"]: (T, 68, 3), torch.int32.
