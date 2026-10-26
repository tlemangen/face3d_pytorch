"""利用二维关键点把 FLAME 网格对齐到输入图像。"""

from collections.abc import Sequence

import torch


ImageSize = int | Sequence[int]


def _parse_image_size(image_size: ImageSize) -> tuple[int, int]:
    if isinstance(image_size, int):
        height = width = image_size
    else:
        if len(image_size) != 2:
            raise ValueError("image_size 应为整数或 (height, width)")
        height, width = int(image_size[0]), int(image_size[1])

    if height <= 1 or width <= 1:
        raise ValueError("image_size 的高度和宽度都必须大于 1")
    return height, width


def landmarks_pixels_to_ndc(
    landmarks: torch.Tensor,
    image_size: ImageSize,
) -> torch.Tensor:
    """将像素坐标转换为 x 向右、y 向上的 NDC 坐标。"""
    if landmarks.shape[-1] < 2:
        raise ValueError("landmarks 的最后一维至少应包含 x 和 y")

    height, width = _parse_image_size(image_size)
    x = 2.0 * landmarks[..., 0] / (width - 1) - 1.0
    y = 1.0 - 2.0 * landmarks[..., 1] / (height - 1)
    return torch.stack((x, y), dim=-1)


def default_ibug68_weights(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """返回相机拟合使用的 68 点权重。"""
    weights = torch.ones(68, device=device, dtype=dtype)
    weights[:17] = 0.45  # 面部轮廓受身份形状和遮挡影响较大
    weights[17:27] = 0.75  # 眉毛
    weights[27:36] = 1.20  # 鼻子
    weights[36:48] = 1.30  # 眼睛
    weights[48:60] = 0.85  # 外唇
    weights[60:68] = 0.70  # 内唇受表情影响最大
    return weights


def _solve_weighted_affine_camera(
    design: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    sqrt_weights = weights.clamp_min(1e-8).sqrt().unsqueeze(-1)
    weighted_design = design * sqrt_weights
    weighted_target = target * sqrt_weights

    normal_matrix = weighted_design.transpose(1, 2) @ weighted_design
    right_hand_side = weighted_design.transpose(1, 2) @ weighted_target

    identity = torch.eye(
        normal_matrix.shape[-1],
        device=normal_matrix.device,
        dtype=normal_matrix.dtype,
    ).unsqueeze(0)
    solution = torch.linalg.solve(normal_matrix + ridge * identity, right_hand_side)
    return solution.transpose(1, 2)


def fit_affine_camera(
    flame_landmarks: torch.Tensor,
    image_landmarks: torch.Tensor,
    image_size: ImageSize,
    *,
    landmark_weights: torch.Tensor | None = None,
    robust_iterations: int = 4,
    huber_delta: float = 0.025,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """拟合从 FLAME 三维坐标到图像 NDC 坐标的稳健仿射相机。

    与仅拟合 ``[scale, tx, ty]`` 的弱透视相机相比，返回的 2x4 相机
    同时利用 FLAME 的深度坐标，可修正残余的 yaw、pitch、roll 和轻微
    的非等比缩放。IRLS-Huber 会降低遮挡或映射误差点的影响。

    Args:
        flame_landmarks: FLAME iBUG 68 三维点，形状为 ``(B, 68, 3)``。
        image_landmarks: 对应图像像素点，形状为 ``(B, 68, 2/3)``。
        image_size: 原图的 ``(height, width)`` 或正方形边长。
        landmark_weights: 可选的关键点基础权重。
        robust_iterations: IRLS 更新次数。
        huber_delta: NDC 中的 Huber 阈值；0.025 约等于 256 图像的 3 像素。
        ridge: 正规方程的数值稳定项。

    Returns:
        形状为 ``(B, 2, 4)`` 的相机矩阵。前三列作用于 ``(x, y, z)``，
        最后一列为二维平移。
    """
    if flame_landmarks.ndim != 3 or flame_landmarks.shape[-1] != 3:
        raise ValueError("flame_landmarks 应为 (B, N, 3)")
    if image_landmarks.ndim != 3 or image_landmarks.shape[-1] < 2:
        raise ValueError("image_landmarks 应为 (B, N, 2/3)")
    if flame_landmarks.shape[:2] != image_landmarks.shape[:2]:
        raise ValueError("两组 landmarks 的批量大小和关键点数量必须一致")
    if flame_landmarks.shape[1] < 4:
        raise ValueError("拟合仿射相机至少需要 4 对关键点")
    if robust_iterations < 0:
        raise ValueError("robust_iterations 不能小于 0")
    if huber_delta <= 0:
        raise ValueError("huber_delta 必须大于 0")

    target_ndc = landmarks_pixels_to_ndc(
        image_landmarks,
        image_size,
    ).to(device=flame_landmarks.device, dtype=flame_landmarks.dtype)

    ones = torch.ones_like(flame_landmarks[..., :1])
    design = torch.cat((flame_landmarks, ones), dim=-1)

    if landmark_weights is None:
        if flame_landmarks.shape[1] == 68:
            base_weights = default_ibug68_weights(
                device=flame_landmarks.device,
                dtype=flame_landmarks.dtype,
            )
        else:
            base_weights = torch.ones(
                flame_landmarks.shape[1],
                device=flame_landmarks.device,
                dtype=flame_landmarks.dtype,
            )
    else:
        if landmark_weights.ndim != 1 or landmark_weights.shape[0] != flame_landmarks.shape[1]:
            raise ValueError("landmark_weights 应为长度等于关键点数量的一维张量")
        base_weights = landmark_weights.to(
            device=flame_landmarks.device,
            dtype=flame_landmarks.dtype,
        )

    weights = base_weights.unsqueeze(0).expand(flame_landmarks.shape[0], -1).clone()
    camera = _solve_weighted_affine_camera(design, target_ndc, weights, ridge)

    for _ in range(robust_iterations):
        projected = torch.einsum("bij,bnj->bni", camera, design)
        residual = torch.linalg.vector_norm(projected - target_ndc, dim=-1)
        robust_weights = torch.where(
            residual <= huber_delta,
            torch.ones_like(residual),
            huber_delta / residual.clamp_min(1e-8),
        )
        weights = base_weights.unsqueeze(0) * robust_weights
        camera = _solve_weighted_affine_camera(design, target_ndc, weights, ridge)

    return camera


def project_with_affine_camera(
    points: torch.Tensor,
    camera: torch.Tensor,
) -> torch.Tensor:
    """使用 ``fit_affine_camera`` 返回的相机投影三维点。"""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points 应为 (B, N, 3)")
    if camera.ndim != 3 or camera.shape[-2:] != (2, 4):
        raise ValueError("camera 应为 (B, 2, 4)")
    if points.shape[0] != camera.shape[0]:
        raise ValueError("points 与 camera 的批量大小必须一致")

    homogeneous_points = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    return torch.einsum("bij,bnj->bni", camera, homogeneous_points)


def smooth_affine_camera_sequence(
    camera: torch.Tensor,
    current_weight: float = 0.65,
) -> torch.Tensor:
    """对按时间排列的相机做指数平滑，减少视频叠加时的抖动。"""
    if camera.ndim != 3 or camera.shape[-2:] != (2, 4):
        raise ValueError("camera 应为 (T, 2, 4)")
    if not 0.0 < current_weight <= 1.0:
        raise ValueError("current_weight 必须位于 (0, 1] 区间")
    if camera.shape[0] <= 1 or current_weight == 1.0:
        return camera

    smoothed = [camera[0]]
    for frame_camera in camera[1:]:
        smoothed.append(
            current_weight * frame_camera
            + (1.0 - current_weight) * smoothed[-1]
        )
    return torch.stack(smoothed, dim=0)
