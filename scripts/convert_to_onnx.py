import torch
import onnx
import argparse
from classification.models.resnet import ResNet
from classification.models.densenet import DenseNet
from segmentation.models.unet import UNet
from segmentation.models.unet_plus_plus import UNetPlusPlus

def export_to_onnx(model, input_shape, output_path, device):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    torch.onnx.export(model, dummy_input, output_path, opset_version=11)
    print(f'Model exported to {output_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'densenet', 'unet', 'unet_plus_plus'])
    parser.add_argument('--model_path', type=str, default='saved_models/classification/model.pth')
    parser.add_argument('--output_path', type=str, default='saved_models/model.onnx')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {
        'resnet': ResNet,
        'densenet': DenseNet,
        'unet': UNet,
        'unet_plus_plus': UNetPlusPlus
    }
    input_shape = (1, 3, 224, 224) if args.model in ['resnet', 'densenet'] else (1, 3, 256, 256)
    model = model_dict[args.model]()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    export_to_onnx(model, input_shape, args.output_path, device)

if __name__ == '__main__':
    main()