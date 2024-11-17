import pytest
import torch
from classification.models.resnet import ResNet
from classification.utils_classification import get_data_loader

def test_resnet_forward():
    model = ResNet(num_classes=10)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 10)

def test_data_loader():
    data_loader = get_data_loader('data/classification', batch_size=2, train=True)
    inputs, labels = next(iter(data_loader))
    assert inputs.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)# Add unit tests for classification module
