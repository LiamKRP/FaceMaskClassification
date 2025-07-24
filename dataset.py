import os
import cv2
from torch.utils.data import Dataset
import albumentations as albu

class MaskDataset(Dataset):
    """
    Custom dataset for face mask images and labels
    """

    def __init__(self, transform):
        self.root_dir = 'dataset'
        self.classes = ['incorrect_mask', 'with_mask', 'without_mask']
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for i, cls in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')): # if its an image extension
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(i)
            
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, id):
        img_path = self.image_paths[id]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        label = self.labels[id]

        return image, label
        

        
