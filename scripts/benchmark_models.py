import torch
import time
from classification.models.resnet import ResNet
from classification.models.densenet import DenseNet
from segmentation.models.unet import UNet
from segmentation.models.unet_plus_plus import UNetPlusPlus

def benchmark_model(model, input_shape, device, iterations=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            model(dummy_input)
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f'Average inference time: {avg_time:.4f} seconds')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        'resnet': (ResNet(), (1, 3, 224, 224)),
        'densenet': (DenseNet(), (1, 3, 224, 224)),
        'unet': (UNet(), (1, 3, 256, 256)),
        'unet_plus_plus': (UNetPlusPlus(), (1, 3, 256, 256))
    }
    
    for model_name, (model, input_shape) in models.items():
        print(f'Benchmarking {model_name}...')
        benchmark_model(model, input_shape, device)

if __name__ == '__main__':
    main()# Add model benchmarking script
