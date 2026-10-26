"""
https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/data_preparation.py
"""

import torch
import torch.nn as nn
import face_alignment
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from PIL import Image
from scipy.io import loadmat


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0 : 2 * npts - 1 : 2, 0:3] = x.transpose()
    A[0 : 2 * npts - 1 : 2, 3] = 1

    A[1 : 2 * npts : 2, 4:7] = x.transpose()
    A[1 : 2 * npts : 2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224.0, mask=None):
    w0, h0 = img.size
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.BICUBIC)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img, lm, mask


# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack(
        [
            lm[lm_idx[0], :],
            np.mean(lm[lm_idx[[1, 2]], :], 0),
            np.mean(lm[lm_idx[[3, 4]], :], 0),
            lm[lm_idx[5], :],
            lm[lm_idx[6], :],
        ],
        axis=0,
    )
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


# utils for face reconstruction
def align_img(img, lm, lm3D, mask=None, target_size=224.0, rescale_factor=102.0):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor / s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    # trans_params = np.array([w0, h0, s, t[0], t[1]])
    trans_params = np.array([w0, h0, s, t[0, 0], t[1, 0]])

    return trans_params, img_new, lm_new, mask_new


def load_lm3d(similarity_Lm3D_all_mat):
    Lm3D = loadmat(similarity_Lm3D_all_mat)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack(
        [
            Lm3D[lm_idx[0], :],
            np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
            np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
            Lm3D[lm_idx[5], :],
            Lm3D[lm_idx[6], :],
        ],
        axis=0,
    )
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


class Deep3DFaceReconAlign(nn.Module):
    """Deep3DFaceRecon 的人脸对齐预处理

    Args:
        similarity_lm3d_all_mat (str): similarity_Lm3D_all.mat 文件的路径
        mask (ndarray, optional): 是与原始图像尺寸一致的有效区域掩码，会和图像执行同步的对齐操作. Defaults to None.
        target_size (float, optional): 对齐后的目标尺寸. Defaults to 224.0.
        rescale_factor (float, optional): 决定对齐后人脸在 224x224 图像中所占的尺度. Defaults to 102.0.
    """

    def __init__(self, similarity_lm3d_all_mat, mask=None, target_size=224.0, rescale_factor=102.0):
        super().__init__()
        self.lm3d_std = load_lm3d(similarity_lm3d_all_mat)

        # 使用 FaceAlignment 替代 landmark68_detector.pb
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.mask = mask
        self.target_size = target_size
        self.rescale_factor = rescale_factor

    def forward(self, images):
        # images: (B, C, H, W)
        images_aligned = []
        landmarks_aligned = []
        trans_params = []
        for image in images:
            image_pil = TVF.to_pil_image(image)  # (W, H)
            image_np = (255 * image.permute(1, 2, 0)).to(torch.uint8).detach().cpu().numpy()  # (H, W, 3)
            landmarks_np = self.fa.get_landmarks(image_np)
            landmark_np = landmarks_np[0]

            W, H = image_pil.size
            landmark_np[:, -1] = H - 1 - landmark_np[:, -1]

            trans_params_np, image_pil_aligned, landmark_np_aligned, masked_aligned = align_img(
                image_pil,
                landmark_np,
                self.lm3d_std,
                mask=self.mask,
                target_size=self.target_size,
                rescale_factor=self.rescale_factor,
            )
            image_aligned = (TVF.pil_to_tensor(image_pil_aligned) / 255).to(torch.float32)
            landmark_aligned = torch.from_numpy(landmark_np_aligned).to(torch.float32)
            if masked_aligned is not None:
                pass  # NOTE: 暂未实现

            images_aligned.append(image_aligned)
            landmarks_aligned.append(landmark_aligned)
            trans_params.append(torch.from_numpy(trans_params_np).to(torch.float32))
        images_aligned = torch.stack(images_aligned, dim=0).to(images.device)
        landmarks_aligned = torch.stack(landmarks_aligned, dim=0).to(images.device)
        align_params = torch.stack(trans_params, dim=0).to(images.device)

        # aligned_images: (B, C, target_size, target_size)
        # aligned_landmarks: (B, 68, 2)
        # align_params: (B, 5)  缩放裁剪/对齐参数
        #  w0: 原图像宽度
        #  h0: 原图像高度
        #  s: 对齐和裁剪使用的缩放比例
        #  t0: 人脸中心的横向位置
        #  t1: 人脸中心的纵向位置
        align_dict = {"aligned_images": images_aligned, "aligned_landmarks": landmarks_aligned, "align_params": align_params}
        return align_dict


def deep3dfacerecon_inverse_align(
    aligned_tensors: torch.Tensor, align_params: torch.Tensor, interpolation: str = "bilinear"
) -> torch.Tensor:
    """将对齐后的张量恢复到原始图像坐标系。

    Args:
        aligned_tensors:
            对齐坐标系中的张量，形状为 (B, C, H_aligned, W_aligned)。
        align_params:
            对齐参数，形状为 (B, 5)，每行为
            (original_width, original_height, scale, tx, ty)。
        interpolation:
            恢复原始尺寸时使用的插值方式。

    Returns:
        原始图像坐标系中的张量，形状为
        (B, C, original_height, original_width)。

    Raises:
        ValueError:
            当一个 batch 中的原始图像尺寸不一致时抛出。
    """
    if aligned_tensors.ndim != 4:
        raise ValueError("aligned_tensors 必须是形状为 (B, C, H, W) 的张量")

    if align_params.ndim != 2 or align_params.shape[1] != 5:
        raise ValueError("align_params 必须是形状为 (B, 5) 的张量")

    if aligned_tensors.shape[0] != align_params.shape[0]:
        raise ValueError("aligned_tensors 和 align_params 的 batch 大小必须一致")

    batch_size, _, aligned_height, aligned_width = aligned_tensors.shape

    restored_tensors = []
    expected_original_size = None

    for batch_index in range(batch_size):
        original_width, original_height, scale, tx, ty = align_params[batch_index].detach().cpu().tolist()

        original_width = int(round(original_width))
        original_height = int(round(original_height))

        if scale <= 0:
            raise ValueError(f"第 {batch_index} 个样本的 scale 必须大于 0")

        current_original_size = (original_height, original_width)
        if expected_original_size is None:
            expected_original_size = current_original_size
        elif current_original_size != expected_original_size:
            raise ValueError("当前函数要求同一个 batch 中的原始图像尺寸一致")

        # 与 resize_n_crop_img 中的整数转换保持一致。
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)

        left = int(scaled_width / 2 - aligned_width / 2 + (tx - original_width / 2) * scale)
        top = int(scaled_height / 2 - aligned_height / 2 + (original_height / 2 - ty) * scale)

        # 创建缩放后原图尺寸的空画布。
        scaled_canvas = aligned_tensors.new_zeros((1, aligned_tensors.shape[1], scaled_height, scaled_width))

        # 裁剪区域可能超出画布，所以需要计算有效交集。
        destination_left = max(left, 0)
        destination_top = max(top, 0)
        destination_right = min(left + aligned_width, scaled_width)
        destination_bottom = min(top + aligned_height, scaled_height)

        if destination_right > destination_left and destination_bottom > destination_top:
            source_left = destination_left - left
            source_top = destination_top - top
            source_right = source_left + destination_right - destination_left
            source_bottom = source_top + destination_bottom - destination_top

            scaled_canvas[:, :, destination_top:destination_bottom, destination_left:destination_right] = aligned_tensors[
                batch_index : batch_index + 1, :, source_top:source_bottom, source_left:source_right
            ]

        interpolate_arguments = {"input": scaled_canvas, "size": (original_height, original_width), "mode": interpolation}

        if interpolation in {"linear", "bilinear", "bicubic", "trilinear"}:
            interpolate_arguments["align_corners"] = False

        restored_tensor = F.interpolate(**interpolate_arguments)
        restored_tensors.append(restored_tensor)

    return torch.cat(restored_tensors, dim=0)
