import torch
import face_alignment
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import estimate_transform, warp
from loguru import logger


class FAN(object):
    def __init__(self):
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'


class DECAAlign(nn.Module):

    def __init__(self):
        super().__init__()
        self.face_detector = FAN()
        self.target_size = 224

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    @staticmethod
    def _bbox2point(left, right, top, bottom, type='bbox'):
        '''bbox from detector and landmarks are different'''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def forward(self, images):
        # images: (B, C, H, W) tensor
        h, w = images.shape[-2:]
        images_np = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()

        dst_images = []
        tform_params = []
        for image in images_np:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
            else:
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]
            old_size, center = self._bbox2point(left, right, top, bottom, type=bbox_type)
            scale = 1.25
            size = int(old_size * scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )

            DST_PTS = np.array([[0, 0], [0, self.target_size - 1], [self.target_size - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.0

            dst_image = warp(image, tform.inverse, output_shape=(self.target_size, self.target_size))
            dst_image = dst_image.transpose(2, 0, 1)

            dst_images.append(torch.from_numpy(dst_image))
            tform_params.append(torch.from_numpy(tform.params))
        aligned_images = torch.stack(dst_images, dim=0).to(images.device, torch.float32)  # (B, 3, 224, 224)
        align_params = torch.stack(tform_params, dim=0).to(images.device, torch.float32)  # (B, 3, 3)
        align_params = torch.inverse(align_params).transpose(1, 2)

        # aligned_images: (B, C, target_size, target_size)
        # align_params: (B, 3, 3)
        align_dict = {"aligned_images": aligned_images, "align_params": align_params}
        return align_dict


def deca_inverse_align(
    aligned_tensors: torch.Tensor,
    align_params: torch.Tensor,
    interpolation: str = "bilinear",
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """将对齐后的张量变换回原始图像坐标系。

    ``forward`` 返回的 ``align_params`` 采用行向量约定，即对齐图像中的
    像素坐标 ``[x, y, 1]`` 右乘该矩阵后得到原图中的像素坐标。这里为输出
    图像的每个像素计算它在对齐图像中的反向采样位置，再使用
    :func:`torch.nn.functional.grid_sample` 完成可微分采样。

    Args:
        aligned_tensors:
            对齐坐标系中的张量，形状为 ``(B, C, H, W)``。
        align_params:
            ``forward`` 返回的变换矩阵，形状为 ``(B, 3, 3)``。
        interpolation:
            插值方式，可选 ``"nearest"``、``"bilinear"`` 或
            ``"bicubic"``。
        output_size:
            输出的 ``(height, width)``。变换矩阵本身不包含原图尺寸，因此
            需要恢复到不同尺寸的原图时必须显式传入；省略时保持输入张量
            的空间尺寸。

    Returns:
        原始图像坐标系中的张量，形状为
        ``(B, C, output_height, output_width)``。原图范围之外的区域填充为零。

    Raises:
        ValueError:
            输入形状、插值方式或变换矩阵不合法时抛出。
    """
    if aligned_tensors.ndim != 4:
        raise ValueError("aligned_tensors 必须是形状为 (B, C, H, W) 的张量")

    if not aligned_tensors.is_floating_point():
        raise ValueError("aligned_tensors 必须是浮点张量")

    if align_params.ndim != 3 or align_params.shape[1:] != (3, 3):
        raise ValueError("align_params 必须是形状为 (B, 3, 3) 的张量")

    if aligned_tensors.shape[0] != align_params.shape[0]:
        raise ValueError("aligned_tensors 和 align_params 的 batch 大小必须一致")

    supported_interpolations = {"nearest", "bilinear", "bicubic"}
    if interpolation not in supported_interpolations:
        supported = "、".join(sorted(supported_interpolations))
        raise ValueError(f"interpolation 必须是以下值之一：{supported}")

    _, _, aligned_height, aligned_width = aligned_tensors.shape
    if aligned_height <= 0 or aligned_width <= 0:
        raise ValueError("aligned_tensors 的空间尺寸必须大于 0")

    if output_size is None:
        output_height, output_width = aligned_height, aligned_width
    else:
        if not isinstance(output_size, (tuple, list, torch.Size)) or len(output_size) != 2:
            raise ValueError("output_size 必须是 (height, width)")
        output_height, output_width = output_size
        if not isinstance(output_height, int) or not isinstance(output_width, int):
            raise ValueError("output_size 中的 height 和 width 必须是整数")
        if output_height <= 0 or output_width <= 0:
            raise ValueError("output_size 中的 height 和 width 必须大于 0")

    # grid_sample 要求输入和采样网格位于同一设备并使用相同的浮点类型。
    transforms = align_params.to(device=aligned_tensors.device, dtype=aligned_tensors.dtype)
    if not torch.isfinite(transforms).all():
        raise ValueError("align_params 必须只包含有限数值")

    try:
        # align_params 将 aligned -> original；反向采样需要 original -> aligned。
        inverse_transforms = torch.linalg.inv(transforms)
    except RuntimeError as error:
        raise ValueError("align_params 中包含不可逆的变换矩阵") from error

    y_coordinates = torch.arange(output_height, device=aligned_tensors.device, dtype=aligned_tensors.dtype)
    x_coordinates = torch.arange(output_width, device=aligned_tensors.device, dtype=aligned_tensors.dtype)
    y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    homogeneous_grid = torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1).reshape(1, -1, 3)
    homogeneous_grid = homogeneous_grid.expand(aligned_tensors.shape[0], -1, -1)

    source_coordinates = torch.bmm(homogeneous_grid, inverse_transforms)
    homogeneous_scale = source_coordinates[..., 2:3]
    if torch.any(homogeneous_scale.abs() <= torch.finfo(aligned_tensors.dtype).eps):
        raise ValueError("align_params 将部分输出像素映射到了无穷远处")
    source_coordinates = source_coordinates[..., :2] / homogeneous_scale

    # skimage 的对齐变换以角点像素 0 和 size - 1 为边界，所以这里使用
    # align_corners=True，以保持正向与反向变换的像素坐标约定一致。
    source_x = normalize_pixel_coordinates(source_coordinates[..., 0], aligned_width)
    source_y = normalize_pixel_coordinates(source_coordinates[..., 1], aligned_height)
    sampling_grid = torch.stack((source_x, source_y), dim=-1).reshape(aligned_tensors.shape[0], output_height, output_width, 2)

    return F.grid_sample(aligned_tensors, sampling_grid, mode=interpolation, padding_mode="zeros", align_corners=True)


def normalize_pixel_coordinates(coordinates: torch.Tensor, size: int) -> torch.Tensor:
    """把像素坐标转换为 ``grid_sample`` 使用的 ``[-1, 1]`` 坐标。"""
    if size == 1:
        return torch.zeros_like(coordinates)
    return coordinates * (2.0 / (size - 1)) - 1.0
