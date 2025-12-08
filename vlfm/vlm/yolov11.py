# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional

import numpy as np

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from ultralytics import YOLO
except Exception:
    print("Could not import ultralytics. This is OK if you are only using the client.")


class YOLO11:
    def __init__(self, weights: str = "data/yolov8m.pt", conf_threshold: float = 0.25):
        """Loads the YOLOv8 model (YOLO11 requires PyTorch 2.0+, using v8 for compatibility)."""
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

        # Warm-up
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_image, verbose=False)

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
            iou_thres (float): IOU threshold for NMS.
            classes (list): List of classes to filter by (not used, for compatibility).
            agnostic_nms (bool): Whether to use agnostic NMS (not used, for compatibility).
        """
        # Run inference
        results = self.model(
            image,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )[0]

        # Extract detections
        boxes = results.boxes

        if len(boxes) == 0:
            # No detections
            return ObjectDetections(
                np.zeros((0, 4)), np.zeros(0), [], image_source=image, fmt="xyxy"
            )

        # Convert to normalized coordinates [0, 1]
        xyxy_boxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        h, w = image.shape[:2]
        xyxy_boxes[:, [0, 2]] /= w  # Normalize x coordinates
        xyxy_boxes[:, [1, 3]] /= h  # Normalize y coordinates

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # Map class IDs to COCO class names
        phrases = [COCO_CLASSES[class_id] for class_id in class_ids]

        detections = ObjectDetections(
            xyxy_boxes, confidences, phrases, image_source=image, fmt="xyxy"
        )

        return detections


class YOLO11Client:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/yolo11"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)
        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--weights", type=str, default="data/yolov8m.pt")
    args = parser.parse_args()

    print("Loading YOLOv8 model (using for YOLO11 compatibility)...")

    class YOLO11Server(ServerMixin, YOLO11):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolo11 = YOLO11Server(args.weights)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo11, name="yolo11", port=args.port)
