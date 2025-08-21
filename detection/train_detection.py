import torch
import torch.optim as optim
import yaml
from models.yolo.yolo import YOLO
from models.yolo.yolo_utils import get_yolo_data_loader

def train(model, train_loader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

def main():
    config = yaml.safe_load(open('detection/models/yolo/configs/yolov5.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(num_classes=config['model']['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = get_yolo_data_loader(config['data']['train_dir'], config['training']['batch_size'], train=True)
    
    train(model, train_loader, optimizer, device, config['training']['epochs'])
    torch.save(model.state_dict(), config['model']['weights'])

if __name__ == '__main__':
    main()