import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path 
import torchvision.transforms as transforms 

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(160, 160)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        #self.ids = os.listdir(images_dir)
        self.ids = [file for file in os.listdir(images_dir) if not file.startswith('.')]
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor() 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        filename = self.ids[index]
   
        img_path = self.images_dir / filename
        mask_path = self.masks_dir / filename
        
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
      
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask)
        
        # Maske ist exakt 0.0 oder 1.0
        mask_tensor = (mask_tensor > 0.5).float()
        
        mask_tensor = mask_tensor.squeeze(0) 

        return {
            'image': image_tensor,
            'mask': mask_tensor
        }