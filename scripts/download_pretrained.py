import torch
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
from pathlib import Path

def download_pretrained():
    Path('saved_models/classification').mkdir(parents=True, exist_ok=True)
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    torch.save(resnet.state_dict(), 'saved_models/classification/resnet50.pth')
    densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
    torch.save(densenet.state_dict(), 'saved_models/classification/densenet121.pth')
    # Placeholder: Add YOLO weight download
    print("Pre-trained models downloaded.")

if __name__ == '__main__':
    download_pretrained()