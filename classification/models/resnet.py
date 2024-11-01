import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)# Add ResNet model implementation
