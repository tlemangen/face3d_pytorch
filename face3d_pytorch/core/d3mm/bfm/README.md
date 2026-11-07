# bfm

## Useage

### deep3dfacerecon_bfm

实例化：

```python
from face3d_pytorch.core.d3mm.bfm import Deep3DFaceReconBFM

bfm = Deep3DFaceReconBFM(
    bfm_model_mat="face3d_pytorch/assets/bfm/BFM_model_front.mat",
    # bfm_model_mat="face3d_pytorch/assets/bfm/BFM_model.mat",
)
bfm.cuda()
```

输入身份和表情等参数，返回一个包括顶点和三角面等信息的字典：

```python
mesh_dict = bfm(id, exp, tex, angle, gamma, trans)
```

args:

- id: (B, 80), torch.float32.
- exp: (B, 64), torch.float32.
- tex: (B, 80), torch.float32.
- angle: (B, 3), torch.float32.
- gamma: (B, 27), torch.float32.
- trans: (B, 3), torch.float32.

return:

- mesh_dict["verts"]: (B, 35709, 3), torch.float32. 顶点.
- mesh_dict["faces"]: (70789, 3), torch.int32. 三角面.
- mesh_dict["verts_textures"]: (B, 35709, 3), torch.float32. 不考虑光照的顶点基础颜色.
- mesh_dict["verts_colors"]: (B, 35709, 3), torch.float32. 加入球谐光照后的顶点显示颜色.
- mesh_dict["landmarks"]: (B, 68, 2).
- 
### face3d_bfm

实例化：

```python
from face3d_pytorch.core.d3mm.bfm import Face3DBFM

bfm = Face3DBFM(bfm_mat_path="face3d_pytorch/assets/face3d/BFM.mat", img_size=256, device="cuda")
bfm.cuda()
```

输入身份和表情等参数，返回一个包括顶点和三角面等信息的字典：

```python
mesh_dict = bfm(id_params, exp_params, scale_params, angle_params, trans_params)
```

args:

- id_params: (B, 199), torch.float32.
- exp_params: (B, 29), torch.float32.
- scale_params: (B,), torch.float32.
- angle_params: (B, 3), torch.float32.
- trans_params: (B, 3), torch.float32.

return:

- mesh_dict["verts"]: (B, 53215, 3), torch.float32. 顶点.
- mesh_dict["faces"]: (105840, 3), torch.int32. 三角面.
- mesh_dict["landmarks"]: (B, 68, 2), torch.int32.
