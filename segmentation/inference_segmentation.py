import torch
import cv2
from pathlib import Path
from models.unet import UNet
from models.unet_plus_plus import UNetPlusPlus
from torchvision import transforms

def infer(model, image_path, device):
    model.eval()
    transform = transforms.ToTensor()
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output).cpu().numpy()[0, 0] > 0.5
    return output, img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'unet_plus_plus'])
    parser.add_argument('--image_path', type=str, default='data/segmentation/sample.jpg')
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
    
    image_path = Path(args.image_path)
    mask, img = infer(model, image_path, device)
    # Placeholder: Save or visualize mask
    cv2.imwrite('output_mask.png', mask * 255)

if __name__ == '__main__':
    main()# Add segmentation inference script
