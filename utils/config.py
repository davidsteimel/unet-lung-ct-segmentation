import os
import torch

# hyperparameter
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 30  
NUM_WORKERS = 4 # Number of subprocesses for data loading
PIN_MEMORY = True # Whether to keep data in pinned memory
LOAD_MODEL = False 
TARGET_SIZE = (256, 256)

# Model settings
IN_CHANNELS = 1
NUM_CLASSES = 1 
DROPOUT = 0.3

# Paths
BASE_DIR = "data_processed"
IMAGE_DIR = 'data/images/images/' 
MASK_DIR = 'data/masks/masks/'
LOG_DIR = "results_256"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, str(TARGET_SIZE[1]), "train", "image")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, str(TARGET_SIZE[1]), "train", "mask")
VAL_IMG_DIR = os.path.join(BASE_DIR, str(TARGET_SIZE[1]), "val", "image")
VAL_MASK_DIR = os.path.join(BASE_DIR, str(TARGET_SIZE[1]), "val", "mask")

CHECKPOINT_FILE = "my_checkpoint.pth.tar"


#  ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
# We should rely heavily on data augmentation with limited training data to train the CNN-based segmentation architecture, i.e. U-Net. 
# The various techniques applied to the original dataset for data augmentation are cropping, scaling, rotating, flipping and color manipulation. 
# The ImageDataGenerator function from the Keras library has been used for this purpose.