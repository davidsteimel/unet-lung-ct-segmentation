import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path 
import torchvision.transforms as transforms 

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patient_images = self.get_patient_data(images_dir)
        self.patient_masks = self.get_patient_data(masks_dir)
        self.samples = []
        self.ids = [file for file in os.listdir(images_dir) if not file.startswith('.')]
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor() 
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        for pid in self.patient_images:
            images = self.patient_images[pid]
            num_available_slices = len(images)
            for slice_index in range(num_available_slices):
                self.samples.append((pid, slice_index))
    
    # create dictionary with patient ids as keys and list of (slice_num, path) tuples as values
    def get_patient_data(self, images_dir):
        all_image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
        patient_dict = {}

        for images in all_image_files:
            base_name = os.path.basename(images)
            pid = base_name.rsplit('_', 1)[0]
            slice_num = int(base_name.rsplit('_', 1)[1].split('.')[0])
            
            if pid not in patient_dict:
                patient_dict[pid] = [] 
            patient_dict[pid].append((slice_num, images))
        
        for pid in patient_dict:
            patient_dict[pid].sort()        
        return patient_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        pid, slice_index = self.samples[index]
        img_path = self.patient_images[pid][slice_index][1]
        mask_path = self.patient_masks[pid][slice_index][1]
        
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
      
        image_tensor = self.transform(image)
        image_tensor = (image_tensor - 0.5) / 0.5
        
        mask_tensor = self.mask_transform(mask)
        # mask should be binary
        mask_tensor = (mask_tensor > 0.5).float()
        
        mask_tensor = mask_tensor.squeeze(0) 

        return {
            'image': image_tensor,
            'mask': mask_tensor
        }