# renderer

## Useage

### deep3dfacerecon_nvdiffrast

实例化：

```python
from face3d_pytorch.core.d3mm.flame import FLAMEWrapper

from face3d_pytorch.core.renderer import Deep3DFaceReconNvdiffrast

renderer = Deep3DFaceReconNvdiffrast(center=112.0, focal=1015.0, z_near=5.0, z_far=15.0)
renderer.cuda()
```

输入形状、表情和姿态等参数，返回一个包括顶点和三角面等信息的字典：

```python
render_dict = renderer(verts, faces, verts_colors)
```

args:

- verts: (B, V, 3), torch.float32. 顶点.
- faces: (B, F, 3), torch.int32. 三角面.
- verts_colors: (B, F, 3), torch.float32. 加入球谐光照后的顶点显示颜色.

return:

- render_dict["masks"]: (B, 3, 224, 224), torch.float32. 对齐的面部掩膜.
- render_dict["depths"]: (B, 3, 224, 224), torch.float32. 对齐的深度图.
- render_dict["rendered_images"]: (B, 3, 224, 224), torch.float32. 对齐的渲染面部.
