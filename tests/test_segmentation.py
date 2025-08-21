import pytest
import torch
from detection.models.yolo.yolo import YOLO
from detection.models.yolo.yolo_utils import get_yolo_data_loader

def test_yolo_forward():
    model = YOLO(num_classes=80)
    input_tensor = torch.randn(1, 3, 640, 640)
    output = model(input_tensor)
    assert isinstance(output, list)  # YOLO returns list of detections

def test_yolo_data_loader():
    data_loader = get_yolo_data_loader('data/detection', batch_size=2, train=True)
    inputs, targets = next(iter(data_loader))
    assert inputs.shape == (2, 3, 640, 640)