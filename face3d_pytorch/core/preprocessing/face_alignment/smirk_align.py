import torch
import numpy as np
import torch.nn as nn
import mediapipe as mp
from skimage.transform import estimate_transform, warp
from loguru import logger


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
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


class SMIRKAlign(nn.Module):

    def __init__(self, face_landmarker_path="face3d_pytorch/assets/mediapipe/face_landmarker.task"):
        super().__init__()
        # download model weights from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
        base_options = mp.tasks.BaseOptions(model_asset_path=face_landmarker_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options, num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    def forward(self, images):
        # images: (B, C, H, W), torch.float32

        images_np = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()  # (B, H, W, C), np.uint8
        images_np = np.ascontiguousarray(images_np, dtype=np.uint8)

        aligned_images = []
        tforms = []
        for image_np in images_np:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            height, width = image.height, image.width
            detection_result = self.landmarker.detect(image)
            lmks_3d = detection_result.face_landmarks[0]
            lmks_2d = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_3d)))  # (478, 2)
            lmks_2d[:, 0] = lmks_2d[:, 0] * width
            lmks_2d[:, 1] = lmks_2d[:, 1] * height

            tform = crop_face(image_np, lmks_2d, scale=1.4, image_size=224)
            aligned_image = warp(image_np, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(
                np.uint8
            )  # (224, 224, 3), np.uint8
            tform = np.asarray(tform, dtype=np.float32)  # (3, 3), np.float32

            aligned_images.append(aligned_image)
            tforms.append(tform)
        aligned_images = np.stack(aligned_images, axis=0)  # (B, 224, 224, 3), np.uint8
        tforms = np.stack(tforms, axis=0)  # (B, 3, 3), np.float32

        aligned_images = (
            torch.from_numpy(aligned_images / 255.0).permute(0, 3, 1, 2).to(torch.float32)
        )  # (B, 3, 224, 224), torch.float32
        aligned_images = aligned_images.to(images.device).detach()

        align_dict = {"aligned_images": aligned_images, "align_params": tforms}
        return align_dict


def smirk_inverse_align(aligned_images, tforms, backgrounds):
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
