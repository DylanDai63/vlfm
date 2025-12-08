# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional

import cv2
import numpy as np
import torch

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

sys.path.insert(0, "yolov9/")
try:
    from models.common import DetectMultiBackend  # noqa: E402
    from utils.dataloaders import LoadImages  # noqa: E402
    from utils.general import (  # noqa: E402
        check_img_size,
        non_max_suppression,
        scale_boxes,
    )
    from utils.torch_utils import select_device  # noqa: E402
except Exception:
    print("Could not import yolov9. This is OK if you are only using the client.")
sys.path.pop(0)


class YOLOv9:
    def __init__(self, weights: str, image_size: int = 640, half_precision: bool = True):
        """Loads the YOLOv9 model and saves it to a field."""
        self.device = select_device("")  # Auto-select device
        self.half_precision = self.device.type != "cpu" and half_precision
        self.image_size = image_size

        # Load model
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=self.half_precision)
        self.stride = self.model.stride
        self.image_size = check_img_size(image_size, s=self.stride)  # check img_size
        self.model.warmup(imgsz=(1, 3, self.image_size, self.image_size))  # warmup

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        orig_shape = image.shape[:2]  # (height, width)

        # Preprocess image
        img = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():
            pred = self.model(img, augment=False, visualize=False)

        # Handle model output - may be list or tuple
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # Get first element (predictions)

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
            max_det=1000,
        )[0]

        # Rescale boxes from img_size to original image size
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_shape).round()

        # Normalize coordinates to [0, 1]
        pred[:, 0] /= orig_shape[1]  # x1
        pred[:, 1] /= orig_shape[0]  # y1
        pred[:, 2] /= orig_shape[1]  # x2
        pred[:, 3] /= orig_shape[0]  # y2

        boxes = pred[:, :4]
        logits = pred[:, 4]
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]

        detections = ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")

        return detections


class YOLOv9Client:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yolov9"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--weights", type=str, default="data/yolov9-c.pt")
    args = parser.parse_args()

    print("Loading YOLOv9 model...")

    class YOLOv9Server(ServerMixin, YOLOv9):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolov9 = YOLOv9Server(args.weights)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov9, name="yolov9", port=args.port)
