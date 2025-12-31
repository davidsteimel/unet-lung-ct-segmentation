import os
import torch

# hyperparameter
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10  
NUM_WORKERS = 4 # Number of subprocesses for data loading
PIN_MEMORY = True # Whether to keep data in pinned memory
LOAD_MODEL = False 

# Model settings
IN_CHANNELS = 1
NUM_CLASSES = 1 

# Paths
BASE_DIR = "data_processed"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "image")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "mask")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "image")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val", "mask")

CHECKPOINT_FILE = "my_checkpoint.pth.tar"
