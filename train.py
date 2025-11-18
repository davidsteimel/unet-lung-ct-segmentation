import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

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

         # --- Forward Pass ---
        targets = targets.unsqueeze(1).float()
        predictions = model(data_img)
        bce = loss_fn(predictions, targets)
        dice = dice_loss(torch.sigmoid(predictions), targets, multiclass=False)
        loss = bce + dice

        # --- Backward Pass ---
        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      
        optimizer.step()      

        # --- Update Ladebalken ---
        loop.set_postfix(loss=loss.item())

def main():
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

    print(f"Starte Training auf {config.DEVICE} für {config.NUM_EPOCHS} Epochen...")
    
    scaler = None 

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoche {epoch+1}/{config.NUM_EPOCHS}")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

        val_dice_score = check_accuracy(val_loader, model, device=config.DEVICE)

        scheduler.step(val_dice_score)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=config.DEVICE
        )

if __name__ == "__main__":
    main()

