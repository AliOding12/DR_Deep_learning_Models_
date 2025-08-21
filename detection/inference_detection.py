import torch
import cv2
from pathlib import Path
from models.yolo.yolo import YOLO
from torchvision import transforms

def infer(model, image_path, device):
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640))
    ])
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions, img

def main():
    config = yaml.safe_load(open('detection/models/yolo/configs/yolov5.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO(num_classes=config['model']['num_classes'])
    model.load_state_dict(torch.load(config['model']['weights'], map_location=device))
    model.to(device)
    
    image_path = Path('data/detection/test/sample.jpg')
    predictions, img = infer(model, image_path, device)
    # Placeholder: Process and save predictions
    print(f'Predictions for {image_path}: {predictions}')

if __name__ == '__main__':
    main()