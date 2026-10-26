import cv2
import torch
import collections
import numpy as np
import torch.nn as nn
from skimage.transform import estimate_transform, warp
from loguru import logger

from .tracker.face_tracker import FaceTracker
from .tracker.utils import get_landmarks


def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def crop_face(frame, landmarks, scale=1.0):
    image_size = 224
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform


class SPECTREAlign(nn.Module):

    def __init__(
        self,
        retinaface_resnet50_final_pth="face3d_pytorch/assets/ibug/face_detection/retina_face/weights/Resnet50_Final.pth",
        fan_2dfan2_pth="face3d_pytorch/assets/ibug/face_alignment/fan/weights/2dfan2.pth",
        device="cuda",
    ):
        super().__init__()
        self.face_tracker = FaceTracker(retinaface_resnet50_final_pth, fan_2dfan2_pth, device)

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    @torch.no_grad()
    def forward(self, video):
        # video: (T, C, H, W), torch.float32, RGB

        device = video.device

        video_np = (255 * video.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()  # (T, H, W, C), np.uint8, RGB
        video_np = np.ascontiguousarray(video_np, dtype=np.uint8)
        video_cv = video_np[..., ::-1]  # BGR

        face_info = collections.defaultdict(list)
        for frame_cv in video_cv:
            detected_faces = self.face_tracker.face_detector(frame_cv, rgb=False)  # (1, 15)
            landmarks, scores = self.face_tracker.landmark_detector(frame_cv, detected_faces, rgb=False)  # (1, 68, 2), (1, 68)

            face_info["bbox"].append(detected_faces)
            face_info["landmarks"].append(landmarks)
            face_info["landmarks_scores"].append(scores)

        landmarks = get_landmarks(face_info)  # List[(68, 2)]
        landmarks = landmarks_interpolate(landmarks)

        # pad
        frames_cv = np.concatenate(
            [video_cv[:1], video_cv[:1], video_cv, video_cv[-1:], video_cv[-1:]], axis=0
        )  # (T+4, H, W, C), np.uint8, RGB
        landmarks.insert(0, landmarks[0])
        landmarks.insert(0, landmarks[0])
        landmarks.append(landmarks[-1])
        landmarks.append(landmarks[-1])
        landmarks = np.stack(landmarks, axis=0)  # (T+4, 68, 2)

        L = 50  # chunk size

        indices = list(range(len(frames_cv)))
        overlapping_indices = [indices[i : i + L] for i in range(0, len(indices), L - 4)]

        if len(overlapping_indices[-1]) < 5:
            # if the last chunk has less than 5 frames, pad it with the semilast frame
            overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
            overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
            overlapping_indices = overlapping_indices[:-1]

        aligned_frames_list = []
        tforms_list = []
        for chunk_id in range(len(overlapping_indices)):
            frames_chunk = frames_cv[overlapping_indices[chunk_id]]  # (L, H, W, C), np.uint8, BGR
            landmarks_chunk = landmarks[overlapping_indices[chunk_id]]  # (L, 68, 2)

            aligned_frames = []
            tforms = []
            for frame, kpt in zip(frames_chunk, landmarks_chunk):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tform = crop_face(frame, kpt, scale=1.6)
                aligned_frame = warp(frame, tform.inverse, output_shape=(224, 224))  # (224, 224, C)
                aligned_frames.append(aligned_frame)
                tforms.append(tform)

            aligned_frames = np.stack(aligned_frames, axis=0)  # (L, H, W, C)
            aligned_frames = (
                torch.from_numpy(aligned_frames).permute(0, 3, 1, 2).to(dtype=torch.float32, device=device).detach()
            )  # (L, C, H, W), torch.float32, RGB
            aligned_frames_list.append(aligned_frames)
            tforms = np.stack(tforms, axis=0).astype(dtype=np.float32)  # (L, 3, 3), np.float32
            tforms_list.append(tforms)

        # aligned_frames_list: List[(L, C, H, W)]
        # tforms_list: List[(L, 3, 3)]
        align_dict = {"aligned_images": aligned_frames_list, "align_params": tforms_list}
        return align_dict


def spectre_inverse_align(aligned_images, tforms, backgrounds):
    # aligned_images: (B, C, H, W), torch.float32
    # tforms: (B, 3, 3), np.float32
    # backgrounds: (B, C, H', W'), torch.float32

    device = aligned_images.device

    height, width = backgrounds.shape[-2:]

    aligned_images_np = aligned_images.permute(0, 2, 3, 1).detach().cpu().numpy()  # (B, H, W, C), np.float32
    aligned_images_np = np.ascontiguousarray(aligned_images_np)

    aligned_mask_np = np.ones(aligned_images_np.shape[1:3], dtype=np.float32)

    restored_images_np = []
    restored_masks_np = []
    for aligned_image_np, tform in zip(aligned_images_np, tforms):
        restored_image_np = warp(
            aligned_image_np, inverse_map=tform, output_shape=(height, width), preserve_range=True, mode="constant", cval=0
        )  # (H, W, C), np.float32
        restored_mask_np = warp(
            aligned_mask_np, inverse_map=tform, output_shape=(height, width), preserve_range=True, mode="constant", cval=0
        )  # (H, W), np.float32

        restored_images_np.append(restored_image_np)
        restored_masks_np.append(restored_mask_np)
    restored_images_np = np.stack(restored_images_np, axis=0)  # (B, H, W, C), np.float32
    restored_masks_np = np.stack(restored_masks_np, axis=0)  # (B, H, W), np.float32

    restored_images = torch.from_numpy(restored_images_np).permute(0, 3, 1, 2).to(device)  # (B, C, H, W), torch.float32
    masks = torch.from_numpy(restored_masks_np).unsqueeze(1).to(device)  # (B, 1, H, W), torch.float32
    masks = torch.clamp(masks, min=0, max=1)

    composited = restored_images * masks + backgrounds * (1.0 - masks)

    # composited: (B, C, H, W), torch.float32
    # masks: (B, 1, H, W), torch.float32
    return composited, masks
