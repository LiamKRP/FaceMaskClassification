import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as albu
import os
import json

from dataset import MaskDataset
from model import MaskClassifier

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Function that we will use for training (not evaluation)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _ , preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        return running_loss / total, correct / total
    

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            # no need to do backward or optimizer step
            # because we are not training

            running_loss += loss.item() * images.size(0)
            _ , preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return running_loss / total, correct / total, all_labels, all_preds

def plot_metrics(history, filename):
    if not os.path.exists('metrics'):
        os.mkdir('metrics')

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column
    epochs = range(1, len(history['train_loss']) + 1)

    axs[0].plot(epochs, history['train_acc'], label = 'Train Acc')
    axs[0].plot(epochs, history['val_acc'], label='Val Acc')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Accuracy')

    axs[1].plot(epochs, history['train_loss'], label='Train Loss')
    axs[1].plot(epochs, history['val_loss'], label='Val Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title('Loss')

    plt.tight_layout()
    plt.savefig(f"metrics/metrics_{filename}.png")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Mask Classifier')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if  torch.backends.mps.is_available():
        device = torch.device('mps')
        print('MPS available, using mps')
    
    args = parser.parse_args()

    checkpoint_name = 'mask_classifier.pth'

    transforms = albu.Compose([
        albu.Resize(128, 128),
        # Divide by 255 by setting max_pixel_value=255, but keep mean=0,std=1 so no shift:
        albu.Normalize(mean=(0.0, 0.0, 0.0),
              std=(1.0, 1.0, 1.0),
              max_pixel_value=255.0),
        albu.pytorch.transforms.ToTensorV2(),  # moves HWC→CHW and casts to torch.float32
])

    dataset = MaskDataset(transform=transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MaskClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr_steps = [50, 100, 150]

    scheduler = MultiStepLR(optimizer, milestones =lr_steps, gamma=0.5)


    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, args.epochs +1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, all_labels, all_preds = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch} / {args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
    # Save model
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(model.state_dict(), f'checkpoints/{checkpoint_name}')

    # save history
    if not os.path.exists('history'):
        os.mkdir('history')
    
    if os.path.exists(f"history/history_{checkpoint_name}.json"):
        print(f"history/history_{checkpoint_name}.json file already exists. Overwriting")

    with open(f"history/history_{checkpoint_name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


    # Final evaluation
    print("Classification report")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

    plot_metrics(history, checkpoint_name)

if __name__ == '__main__':
    main()
    
