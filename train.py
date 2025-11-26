import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

import utils.config as config
from utils.dice_score import dice_loss
from unet.unet_model import UNet
from utils.data_loading import BasicDataset
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    
    for batch_idx, data in enumerate(loop):
        data_img = data['image'].to(config.DEVICE)
        targets = data['mask'].to(config.DEVICE)

        #Forward Pass
        targets = targets.unsqueeze(1).float()
        predictions = model(data_img)
        bce = loss_fn(predictions, targets) #Binary Cross-Entropy, misst Pixel-weise Wahrscheinlichkeiten 
        dice = dice_loss(torch.sigmoid(predictions).squeeze(1), targets.squeeze(1), multiclass=False)
        loss = bce + dice

        # Backward Pass
        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      
        optimizer.step()      

        loop.set_postfix(loss=loss.item())

def main():
    parser = argparse.ArgumentParser(description='UNet Training Script')
    
    parser.add_argument('--epochs', '-e', type=int, default=config.NUM_EPOCHS,
                        help='Anzahl der Trainings-Epochen')
    parser.add_argument('--batch-size', '-b', type=int, default=config.BATCH_SIZE,
                        help='Batch Größe für das Training')
    parser.add_argument('--lr', '--learning-rate', type=float, default=config.LEARNING_RATE,
                        help='Lernrate für den Optimierer')
    parser.add_argument('--load', '-l', action='store_true', default=config.LOAD_MODEL,
                        help='Checkpoint laden, um Training fortzusetzen')
    
    args = parser.parse_args()

    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.LOAD_MODEL = args.load

    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES).to(config.DEVICE)

    
    loss_fn = nn.BCEWithLogitsLoss() 
    #optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    optimizer = optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    train_ds = BasicDataset(
        images_dir=config.TRAIN_IMG_DIR,
        masks_dir=config.TRAIN_MASK_DIR,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )

    val_ds = BasicDataset(
        images_dir=config.VAL_IMG_DIR,
        masks_dir=config.VAL_MASK_DIR,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model)

    print(f"Starte Training auf {config.DEVICE} für {config.NUM_EPOCHS} Epochen \n"
          f"mit einer Lernrate von {config.LEARNING_RATE} und einer Batch-Größe von {config.BATCH_SIZE}.")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU-Speicher reserviert: {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
    else:
        print("Training auf CPU")
    
    scaler = None 

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoche {epoch+1}/{config.NUM_EPOCHS}")
        
        train_fn(train_loader, model, optimizer, loss_fn)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

        val_dice_score = check_accuracy(val_loader, model, device=config.DEVICE)

        scheduler.step(val_dice_score)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=config.DEVICE, num_examples=8, epoche=epoch
        )

if __name__ == "__main__":
    main()