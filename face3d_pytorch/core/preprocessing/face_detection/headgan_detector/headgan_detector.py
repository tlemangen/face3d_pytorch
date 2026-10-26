import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2.functional as TVF
from facenet_pytorch import MTCNN, extract_face
from loguru import logger


def check_boxes(boxes, img_size, dst_threshold=0.35, height_recentre=0.0):
    # Check if there are None boxes.
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = next((item for item in boxes[i + 1 :] if item is not None), boxes[i - 1])
    if boxes[0] is None:
        print('Not enough boxes detected.')
        return False, [None]
    boxes = [box[0] for box in boxes]
    # Check if detected faces are very far from each other. Check distances between all boxes.
    maxim_dst = 0
    for i in range(len(boxes) - 1):
        for j in range(len(boxes) - 1):
            dst = max(abs(boxes[i] - boxes[j])) / img_size
            if dst > maxim_dst:
                maxim_dst = dst
    if maxim_dst > dst_threshold:
        print('L_inf distance between bounding boxes %.4f larger than threshold' % maxim_dst)
        return False, [None]
    # Get average box
    avg_box = np.median(boxes, axis=0)
    # Make boxes square.
    offset_w = avg_box[2] - avg_box[0]
    offset_h = avg_box[3] - avg_box[1]
    offset_dif = (offset_h - offset_w) / 2
    # width
    avg_box[0] = avg_box[2] - offset_w - offset_dif
    avg_box[2] = avg_box[2] + offset_dif
    # height - center a bit lower
    avg_box[3] = avg_box[3] + height_recentre * offset_h
    avg_box[1] = avg_box[3] - offset_h
    return avg_box


class HeadGANDetector(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        self.image_size = 256
        self.margin = 100
        self.mtcnn = MTCNN(image_size=self.image_size, margin=self.margin, post_process=False, device=device).eval()
        
        logger.info(f"[{self.__class__.__name__}] Loading done!")

    @torch.no_grad()
    def forward(self, images):
        # images: (B, C, H, W), torch.float32

        _, _, H, W = images.shape
        img_size = int((H + W) / 2)

        bboxes = []
        images_pil = []
        for image in images:
            image_pil = TVF.to_pil_image(image)
            boxes, _ = self.mtcnn.detect(image_pil)  # (N, 4)
            bboxes.append(boxes)
            images_pil.append(image_pil)

        avg_box = check_boxes(bboxes, img_size, dst_threshold=0.35, height_recentre=0.0)

        faces = []
        for image_pil in images_pil:
            face = extract_face(
                image_pil, avg_box, image_size=self.image_size, margin=self.margin
            )  # (3, 256, 256), torch.float32, [0., 255.]
            faces.append(face)
        return faces
