import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def visualize_classification(image_path, prediction, label):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f'Prediction: {prediction}, Label: {label}')
    plt.axis('off')
    plt.show()

def visualize_detection(image_path, boxes):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize_segmentation(image_path, mask):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.show()

def main():
    # Placeholder: Example visualization
    image_path = Path('data/classification/test/sample.jpg')
    visualize_classification(image_path, 'class1', 'class1')
    visualize_detection(image_path, [[50, 50, 100, 100]])
    visualize_segmentation(image_path, cv2.imread('output_mask.png', cv2.IMREAD_GRAYSCALE))

if __name__ == '__main__':
    main()# Add visualization script for results
