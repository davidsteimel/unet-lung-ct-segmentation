import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = os.listdir(images_dir)
        
        print(f'Dataset erstellt mit {len(self.ids)} Beispielen.')
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        filename = self.ids[index]
        
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
       
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        image_tensor = torch.from_numpy(np.asarray(image))
        mask_tensor = torch.from_numpy(np.asarray(mask))
        
        # Normalisierung (Werte 0-255 -> 0.0-1.0)
        image_tensor = image_tensor.float() / 255.0
        mask_tensor = mask_tensor.float() / 255.0
    
        return {
            'image': image_tensor,
            'mask': mask_tensor
        }