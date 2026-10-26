"""MediaPipe Face Landmarker 到 iBUG/FLAME 68 点的映射。"""

from typing import Final

import torch


# 每一项对应一个 iBUG 68 关键点。包含两个索引时取两点均值。
# 映射来源：PeizhiYan/Mediapipe_2_Dlib_Landmarks。
MEDIAPIPE_TO_IBUG68: Final[tuple[tuple[int, int], ...]] = (
    (127, 127), (234, 234), (93, 93), (132, 58), (58, 172), (136, 136),
    (150, 150), (176, 176), (152, 152), (400, 400), (379, 379),
    (365, 365), (397, 288), (288, 361), (323, 323), (454, 454),
    (356, 356), (70, 70), (63, 63), (105, 105), (66, 66), (107, 107),
    (336, 336), (296, 296), (334, 334), (293, 293), (300, 300),
    (168, 6), (197, 195), (5, 5), (4, 4), (75, 75), (97, 97), (2, 2),
    (326, 326), (305, 305), (33, 33), (160, 160), (158, 158),
    (133, 133), (153, 153), (144, 144), (362, 362), (385, 385),
    (387, 387), (263, 263), (373, 373), (380, 380), (61, 61), (39, 39),
    (37, 37), (0, 0), (267, 267), (269, 269), (291, 291), (321, 321),
    (314, 314), (17, 17), (84, 84), (91, 91), (78, 78), (82, 82),
    (13, 13), (312, 312), (308, 308), (317, 317), (14, 14), (87, 87),
)


def mediapipe_to_ibug68(landmarks: torch.Tensor) -> torch.Tensor:
    """把 MediaPipe 478 点转换为与 FLAME 对应的 iBUG 68 点。

    Args:
        landmarks: 形状为 ``(..., 478, D)`` 的张量，``D`` 可以为 2 或 3。

    Returns:
        形状为 ``(..., 68, D)`` 的张量。
    """
    if landmarks.ndim < 2:
        raise ValueError("landmarks 至少应包含关键点和坐标两个维度")
    if landmarks.shape[-2] < 468:
        raise ValueError(
            f"MediaPipe landmarks 至少应包含 468 点，实际为 {landmarks.shape[-2]} 点"
        )

    correspondence = torch.as_tensor(
        MEDIAPIPE_TO_IBUG68,
        device=landmarks.device,
        dtype=torch.long,
    )
    paired_landmarks = landmarks[..., correspondence, :]
    return paired_landmarks.mean(dim=-2)
