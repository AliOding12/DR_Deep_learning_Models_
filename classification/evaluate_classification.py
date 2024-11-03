import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import argparse
from models.alexnet import AlexNet
from models.vgg import VGG16
from models.resnet import ResNet
from models.densenet import DenseNet
from utils_classification import get_data_loader, load_model

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    accuracy = accuracy_score(targets, predictions)
    print(f'Test Loss: {running_loss / len(test_loader)}, Accuracy: {accuracy}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['alexnet', 'vgg', 'resnet', 'densenet'])
    parser.add_argument('--data_dir', type=str, default='data/classification')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', type=str, default='saved_models/classification/model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {
        'alexnet': AlexNet,
        'vgg': VGG16,
        'resnet': ResNet,
        'densenet': DenseNet
    }
    model = model_dict[args.model](num_classes=10)
    model = load_model(model, args.model_path, device)
    criterion = nn.CrossEntropyLoss()
    test_loader = get_data_loader(args.data_dir, args.batch_size, train=False)
    
    evaluate(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()# Add classification evaluation script
