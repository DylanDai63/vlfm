# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""
Simplified YOLOv9 wrapper that loads models directly with PyTorch,
avoiding the import issues with YOLOv9's DetectMultiBackend.
"""

import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    """Non-Maximum Suppression (NMS) on inference results."""

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """Clip boxes (xyxy) to image shape (height, width)."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


class YOLOv9:
    def __init__(self, weights: str, image_size: int = 640, half_precision: bool = True):
        """Loads the YOLOv9 model directly with PyTorch."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half_precision = self.device.type != "cpu" and half_precision
        self.image_size = image_size

        # Load model directly with PyTorch
        print(f"Loading weights from {weights}...")
        ckpt = torch.load(weights, map_location=self.device)

        # Extract model
        if "ema" in ckpt:
            self.model = ckpt["ema"].float()
        elif "model" in ckpt:
            self.model = ckpt["model"].float()
        else:
            self.model = ckpt.float()

        # Set to eval mode
        self.model.eval()

        # Convert to half precision if requested
        if self.half_precision:
            self.model.half()

        # Get stride
        try:
            self.stride = int(self.model.stride.max())
        except:
            self.stride = 32

        # Warmup
        print("Warming up...")
        dummy_img = torch.zeros((1, 3, self.image_size, self.image_size)).to(self.device)
        if self.half_precision:
            dummy_img = dummy_img.half()
        with torch.no_grad():
            _ = self.model(dummy_img)
        print("Model ready!")

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
        img = letterbox(image, new_shape=self.image_size, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]

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
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_shape)

        # Normalize coordinates to [0, 1]
        pred[:, 0] /= orig_shape[1]  # x1
        pred[:, 1] /= orig_shape[0]  # y1
        pred[:, 2] /= orig_shape[1]  # x2
        pred[:, 3] /= orig_shape[0]  # y2

        boxes = pred[:, :4].cpu().numpy()
        logits = pred[:, 4].cpu().numpy()
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5].cpu().numpy()]

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

    print("Loading YOLOv9 model (simplified loader)...")

    class YOLOv9Server(ServerMixin, YOLOv9):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolov9 = YOLOv9Server(args.weights)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov9, name="yolov9", port=args.port)
