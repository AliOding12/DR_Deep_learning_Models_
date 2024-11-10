import torch
import torch.nn as nn
import argparse
from models.unet import UNet
from models.unet_plus_plus import UNetPlusPlus
from torchvision import transforms
from torch.utils.data import DataLoader
from segmentation.train_segmentation import SegmentationDataset

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, masks in test_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks.unsqueeze(1))
            running_loss += loss.item()
    print(f'Test Loss: {running_loss / len(test_loader)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet_plus_plus'])
    parser.add_argument('--data_dir', type=str, default='data/segmentation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='saved_models/segmentation/model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {
        'unet': UNet,
        'unet_plus_plus': UNetPlusPlus
    }
    model = model_dict[args.model]()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    transform = transforms.ToTensor()
    test_dataset = SegmentationDataset(args.data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    evaluate(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()# Add segmentation evaluation script
