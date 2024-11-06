import torch
import yaml
from models.yolo.yolo import YOLO
from models.yolo.yolo_utils import get_yolo_data_loader

def evaluate(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()
    print(f'Test Loss: {running_loss / len(test_loader)}')

def main():
    config = yaml.safe_load(open('detection/models/yolo/configs/yolov5.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(config['model']['weights'], map_location=device))
    model.to(device)
    test_loader = get_yolo_data_loader(config['data']['val_dir'], config['training']['batch_size'], train=False)
    
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()# Add detection evaluation script
