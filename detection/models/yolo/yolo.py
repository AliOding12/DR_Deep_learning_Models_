import torch
from torchvision.models.detection import yolov5
from .yolo_utils import load_yolo_weights

class YOLO(torch.nn.Module):
    def __init__(self, num_classes=80, weights_path=None):
        super(YOLO, self).__init__()
        self.model = yolov5(pretrained=weights_path is None)
        if weights_path:
            self.model = load_yolo_weights(self.model, weights_path)

    def forward(self, x):
        return self.model(x)