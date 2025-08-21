import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from models.alexnet import AlexNet
from models.vgg import VGG16
from models.resnet import ResNet
from models.densenet import DenseNet
from utils_classification import get_data_loader, save_model

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['alexnet', 'vgg', 'resnet', 'densenet'])
    parser.add_argument('--data_dir', type=str, default='data/classification')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved_models/classification/model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {
        'alexnet': AlexNet,
        'vgg': VGG16,
        'resnet': ResNet,
        'densenet': DenseNet
    }
    model = model_dict[args.model](num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = get_data_loader(args.data_dir, args.batch_size, train=True)
    
    train(model, train_loader, criterion, optimizer, device, args.epochs)
    save_model(model, args.save_path)

if __name__ == '__main__':
    main()