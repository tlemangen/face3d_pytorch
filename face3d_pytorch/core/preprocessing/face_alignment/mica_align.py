import cv2
import torch
import numpy as np
import torch.nn as nn
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from loguru import logger


class MICAAlign(nn.Module):

    def __init__(self):
        super().__init__()
        self.app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(224, 224))

        self.input_mean = 127.5
        self.input_std = 127.5

        logger.info(f"[{self.__class__.__name__}] Loading done!")

    @torch.no_grad()
    def forward(self, images):
        # images: (B, C, H, W), torch.float32, RGB

        device = images.device

        images_np = (255 * images.permute(0, 2, 3, 1)).to(torch.uint8).detach().cpu().numpy()  # (B, H, W, C), np.uint8, RGB
        images_np = np.ascontiguousarray(images_np, dtype=np.uint8)
        images_cv = images_np[..., ::-1]  # BGR

        aligned_images_112x112 = []
        for image_cv in images_cv:
            faces = self.app.get(image_cv)
            if len(faces) == 0:
                logger.error(f"[{self.__class__.__name__}] No face detected!")
            else:
                kps = faces[0].kps  # (5, 2), np.float32

            aligned_image_cv_112x112 = face_align.norm_crop(image_cv, landmark=kps)
            blob = cv2.dnn.blobFromImages(
                [aligned_image_cv_112x112],
                1.0 / self.input_std,
                (112, 112),
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            )
            aligned_image_112x112 = torch.from_numpy(blob[0]).to(torch.float32)  # (C, 112, 112), torch.float32, RGB, normalized

            aligned_images_112x112.append(aligned_image_112x112)

        aligned_images_112x112 = torch.stack(aligned_images_112x112, dim=0)  # (B, C, 112, 112), torch.float32, RGB

        aligned_images_112x112 = aligned_images_112x112.to(device).detach()
        
        align_dict = {
            "aligned_images": aligned_images_112x112
        }
        return align_dict
