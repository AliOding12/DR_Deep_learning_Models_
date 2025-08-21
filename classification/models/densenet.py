import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(DenseNet, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)