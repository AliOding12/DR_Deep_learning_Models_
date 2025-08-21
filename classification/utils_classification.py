import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cuda'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model