import argparse
import torch
import cv2
import albumentations as albu
import numpy as np
import os
import matplotlib.pyplot as plt

from model import MaskClassifier

def main():
    parser = argparse.ArgumentParser(description='Inference for one image')
    parser.add_argument('--model_path', required=True, help='Path to the save model .pth')
    parser.add_argument('--image_path', required=True, help='Path to the image to do inference on')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available, using as inference device")

    model = MaskClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    transforms = albu.Compose([
        albu.Resize(128, 128),
        # Divide by 255 by setting max_pixel_value=255, but keep mean=0,std=1 so no shift:
        albu.Normalize(mean=(0.0, 0.0, 0.0),
              std=(1.0, 1.0, 1.0),
              max_pixel_value=255.0),
        albu.ToTensorV2(),  # moves HWC→CHW and casts to torch.float32
    ]
    )

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = transforms(image=image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    classes = ['incorrect_mask', 'with_mask', 'without_mask']

    print(f"Prediction: {classes[pred]} (class {pred}) with probabilities {probs}")

    # show the image
    plt.imshow(image)
    plt.title(f"Prediction: {classes[pred]} - Probabilities: {(np.max(probs)*100):.2f} %")
    plt.axis('off')  # optional: hide axis ticks
    plt.show()

if __name__ == '__main__':
    main()