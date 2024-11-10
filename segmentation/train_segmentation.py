import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from models.unet import UNet
from models.unet_plus_plus import UNetPlusPlus
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = list(self.data_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.data_dir / f"{img_path.stem}_mask.png"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, masks in train_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet_plus_plus'])
    parser.add_argument('--data_dir', type=str, default='data/segmentation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved_models/segmentation/model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {
        'unet': UNet,
        'unet_plus_plus': UNetPlusPlus
    }
    model = model_dict[args.model]().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    transform = transforms.ToTensor()
    train_dataset = SegmentationDataset(args.data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    train(model, train_loader, criterion, optimizer, device, args.epochs)
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    main()# Add segmentation training script
