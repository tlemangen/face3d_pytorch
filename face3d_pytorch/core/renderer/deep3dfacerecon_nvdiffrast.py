"""
https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/models/facerecon_model.py
https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/util/nvdiffrast.py
"""

"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""

import torch
import numpy as np
import torch.nn as nn
import nvdiffrast.torch as dr
import torch.nn.functional as F
from typing import List


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array(
        [[n / x, 0, 0, 0], [0, n / -x, 0, 0], [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)], [0, 0, -1, 0]]
    ).astype(np.float32)


class MeshRenderer(nn.Module):
    def __init__(self, rasterize_fov, znear=0.1, zfar=10, rasterize_size=224):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear, f=zfar)).matmul(torch.diag(torch.tensor([1.0, -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.glctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        vertex_ndc = vertex @ ndc_proj.t()
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=device)
            print("create glctx on device cuda:%d" % device.index)

        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vetex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.glctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges)

        depth, _ = dr.interpolate(vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image


class Deep3DFaceReconNvdiffrast(nn.Module):
    """在 Deep3DFaceRecon 中用到的 基于 Nvdiffrast 渲染器
    相机内参矩阵为：
    K = [[f, 0, c],
         [0, f, c],
         [0, 0, 1]]
    其中 f 代表 focal, c 代表 center

    Args:
        center (float, optional): 相机主点坐标，并同时被用来决定渲染图像尺寸，默认值 112 表明原图像是 224 尺寸的. Defaults to 112.0.
        focal (float, optional): 透视相机的焦距，单位是像素. Defaults to 1015.0.
        z_near (float, optional): 近裁剪平面到相机的距离, 只有位于近裁剪平面之后的三维几何才会进入有效渲染范围, 单位是分米. Defaults to 5.0.
        z_far (float, optional): 远裁剪平面到相机的距离, 超过远裁剪平面的几何同样会被裁掉, 单位是分米. Defaults to 15.0.
    """

    def __init__(self, center=112.0, focal=1015.0, z_near=5.0, z_far=15.0):
        super().__init__()
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        self.renderer = MeshRenderer(rasterize_fov=fov, znear=z_near, zfar=z_far, rasterize_size=int(2 * center)).eval()
        
    def forward(self, vertices, trifaces, vertice_colors):
        masks, depths, rendered_images = self.renderer(vertices, trifaces, vertice_colors)
        render_dict = {
            "masks": masks,
            "depths": depths,
            "rendered_images": torch.clamp(rendered_images, min=0., max=1.),
        }
        return render_dict
