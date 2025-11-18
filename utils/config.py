import os
import torch

# Hyperparameter
LEARNING_RATE = 1e-4
DEVICE =  torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16  
NUM_EPOCHS = 10  
NUM_WORKERS = 2  
PIN_MEMORY = True
LOAD_MODEL = False 

# Modell-Einstellungen
IN_CHANNELS = 1
NUM_CLASSES = 1 

# Pfade
BASE_DIR = "data_processed"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "image")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "mask")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "image")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val", "mask")

CHECKPOINT_FILE = "my_checkpoint.pth.tar"
