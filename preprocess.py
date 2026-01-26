import os
import glob
import random
import shutil
import cv2
import numpy as np
import utils.config as config
import json

all_image_files = glob.glob(os.path.join(config.IMAGE_DIR, '*.jpg'))
all_mask_files = glob.glob(os.path.join(config.MASK_DIR, '*.jpg'))

patient_ids = set()

for image_path in all_image_files:
    base_name = os.path.basename(image_path)
    patient_id = base_name.rsplit('_', 1)[0] 
    patient_ids.add(patient_id)

patient_ids = list(patient_ids)
random.seed(42)
random.shuffle(patient_ids)

num_total = len(patient_ids)
train_split_end = int(num_total * 0.7)
val_split_end = int(num_total * 0.8)

train_ids = set(patient_ids[:train_split_end])
val_ids = set(patient_ids[train_split_end:val_split_end])
test_ids = set(patient_ids[val_split_end:])

split_info = {
    "seed": 42,
    "train_ids": sorted(list(train_ids)),
    "val_ids": sorted(list(val_ids)),
    "test_ids": sorted(list(test_ids))
}

info_file_path = os.path.join(config.OUTPUT_DIR, str(config.TARGET_SIZE[1]),"split_info.json")
with open(info_file_path, "w") as f:
    json.dump(split_info, f, indent=4)

print(f'Total: {num_total} | Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}')

splits = ['train', 'val', 'test']
types = ['image', 'mask']
resolution = str(config.TARGET_SIZE[1])

for split in splits:
    for type_ in types:
        path = os.path.join('data_processed', resolution, split, type_)
        os.makedirs(path, exist_ok=True)
        print(f"Created/checked directory: {path}")

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
        print(f"Invalid format skipped: {base_name}")
        continue

    mask_base_name = f"{patient_id}_mask_{slice_part}" 
    mask_path = os.path.join(config.MASK_DIR, mask_base_name)

    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {base_name}")
        continue

    if patient_id in train_ids:
        split_dir = 'train'
    elif patient_id in val_ids:
        split_dir = 'val'
    elif patient_id in test_ids:
        split_dir = 'test'
    else:
        continue

    # load image as grayscale     
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    if img.shape[:2] == config.TARGET_SIZE:
        img_resized = img
    else:
        img_resized = cv2.resize(img, config.TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    multiclass_mask_color = cv2.imread(mask_path, cv2.IMREAD_COLOR) 
    
    if multiclass_mask_color is None: continue

    lower_blue = np.array([200, 0, 0])
    upper_blue = np.array([255, 50, 50])
    binary_mask = cv2.inRange(multiclass_mask_color, lower_blue, upper_blue)

    if binary_mask.shape[:2] == config.TARGET_SIZE:
        mask_resized = binary_mask
    else:
        mask_resized = cv2.resize(binary_mask, config.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(config.BASE_DIR, resolution, split_dir, 'image', base_name), img_resized)
    cv2.imwrite(os.path.join(config.BASE_DIR, resolution, split_dir, 'mask', base_name), mask_resized)

print("Processing completed!")