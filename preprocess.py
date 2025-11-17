import os
import glob
import random
import shutil
import cv2
import numpy as np


IMAGE_DIR = 'data/images/images/' 
MASK_DIR = 'data/masks/masks/'

all_image_files = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
all_mask_files = glob.glob(os.path.join(MASK_DIR, '*.jpg'))

patient_ids = set()

for image_path in all_image_files:
    base_name = os.path.basename(image_path)
    patient_id = base_name.rsplit('_', 1)[0] 
    patient_ids.add(patient_id)

patient_ids = list(patient_ids)
random.shuffle(patient_ids)

num_total = len(patient_ids)
tarin_split_end = int(num_total * 0.7)
val_split_end = int(num_total * 0.8)

train_ids = set(patient_ids[:tarin_split_end])
val_ids = set(patient_ids[tarin_split_end:val_split_end])
test_ids = set(patient_ids[val_split_end:])

print(f'Total patients: {num_total}')
print(f'Training patients: {len(train_ids)}')
print(f'Validation patients: {len(val_ids)}')
print(f'Testing patients: {len(test_ids)}')


for image_path in all_image_files:
    base_name = os.path.basename(image_path)
    patient_id = base_name.rsplit('_', 1)[0] 

    try:
        slice_part = base_name.rsplit('_', 1)[1] 
    except IndexError:
        print(f"Fehlerhaftes Format übersprungen: {base_name}")
        continue

    mask_base_name = f"{patient_id}_mask_{slice_part}" 
    mask_path = os.path.join(MASK_DIR, mask_base_name)

    if not os.path.exists(mask_path):
        print(f"Warnung: Maske nicht gefunden für {base_name}")
        continue

    if patient_id in train_ids:
            target_img_dir = 'data_processed/train/image/'
            target_mask_dir = 'data_processed/train/mask/'
    elif patient_id in val_ids:
            target_img_dir = 'data_processed/val/image/'
            target_mask_dir = 'data_processed/val/mask/'
    elif patient_id in test_ids:
            target_img_dir = 'data_processed/test/image/'
            target_mask_dir = 'data_processed/test/mask/'
    else:
        continue

    shutil.copy(image_path, os.path.join(target_img_dir, base_name))

    multiclass_mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
    
    if multiclass_mask_color is None:
        print(f"Warnung: Maske konnte nicht geladen werden {mask_path}")
        continue
        
    LUNGEN_FARBE_BGR = np.array([254, 0, 0]) 

    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    
    binary_mask = cv2.inRange(multiclass_mask_color, lower_blue, upper_blue)
    
    cv2.imwrite(os.path.join(target_mask_dir, base_name), binary_mask)

print("Verarbeitung abgeschlossen!")
