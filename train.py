import torch
import csv
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dice_score import TopKDiceLoss
from tqdm import tqdm
import argparse
import utils.config as config
from unet.unet_model import UNet
from utils.data_loading import BasicDataset, get_dataloader
from utils.utils import (
    check_accuracy,
    save_predictions_as_imgs,
    evaluate
)
from perun import monitor

def train_fn(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0
    total_samples = 0
    
    for batch_idx, data in enumerate(loop):
        data_img = data['image'].to(config.DEVICE)
        targets = data['mask'].to(config.DEVICE)

        #Forward Pass
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        
        predictions = model(data_img)
        loss = loss_fn(predictions, targets)

        # Backward Pass
        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)      
        optimizer.step()      

        batch_size = data_img.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
        loop.set_postfix(loss=loss.item())

    return running_loss / total_samples

def main():
    parser = argparse.ArgumentParser(description='UNet Training Script')
    
    parser.add_argument('--epochs', '-e', type=int, default=config.NUM_EPOCHS,
                        help='Num of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', '--learning-rate', type=float, default=config.LEARNING_RATE,
                        help='Learning rate for the optimizer')
    parser.add_argument('--load', '-l', action='store_true', default=config.LOAD_MODEL,
                        help='Load checkpoint to resume training')
    
    args = parser.parse_args()

    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.LOAD_MODEL = args.load

    model = UNet(n_channels=config.IN_CHANNELS, n_classes=config.NUM_CLASSES, dropout_p=config.DROPOUT_P).to(config.DEVICE)
    loss_fn = TopKDiceLoss(k=20, smooth=1e-5) 

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4, 
        betas = (0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    train_ds = BasicDataset(
        images_dir=config.TRAIN_IMG_DIR,
        masks_dir=config.TRAIN_MASK_DIR,
    )

    train_loader = get_dataloader(
        dataset= train_ds,
        mode="train",
        batch_size=config.BATCH_SIZE
    )

    val_ds = BasicDataset(
        images_dir=config.VAL_IMG_DIR,
        masks_dir=config.VAL_MASK_DIR,
    )

    val_loader = get_dataloader(
        dataset = val_ds,
        mode="val",
        batch_size=config.BATCH_SIZE
    )

    print(f"Starting training on {config.DEVICE} for {config.NUM_EPOCHS} epochs \n"
          f"with a learning rate of {config.LEARNING_RATE} and a batch size of {config.BATCH_SIZE}.")
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = f"log_{config.TARGET_SIZE[1]}_{config.NUM_EPOCHS}_{config.LEARNING_RATE}_unet.csv"
    log_path = os.path.join(config.LOG_DIR, log_file)
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                    'Epoch',
                    'Resolution',
                    'Learning_Rate',
                    'Batch_Size', 
                    'Num_Train_Samples',
                    'Num_Val_Samples',
                    'Train_Loss', 
                    'Val_Loss',
                    'Val_Loss_All',
                    'Train_Dice',
                    'Val_Dice', 
                    'Duration_Sec',
                    'Train_TP',
                    'Train_FP',
                    'Train_TN',
                    'Train_FN',
                    'Train_Precision',
                    'Train_Recall',
                    'Val_TP',
                    'Val_FP',
                    'Val_TN',
                    'Val_FN',
                    'Val_Precision',
                    'Val_Recall'
                ])

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        start_time = time.time()

        train_loss = train_fn(train_loader, model, optimizer, loss_fn)

        val_metrics = evaluate(val_loader, model, loss_fn, device=config.DEVICE)
        train_metrics = evaluate(train_loader, model, loss_fn, device=config.DEVICE)

        scheduler.step(val_metrics['Dice'])
        duration = time.time() - start_time

        train_dice = train_metrics["Dice"]
        val_loss = val_metrics["Loss"]
        val_dice = val_metrics["Dice"]
        val_loss_all = val_metrics["Loss_All"]

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                int(epoch),
                int(config.TARGET_SIZE[1]),
                config.LEARNING_RATE,
                config.BATCH_SIZE,
                len(train_loader.dataset),
                len(val_loader.dataset),
                round(float(train_loss), 4),
                round(float(val_loss), 4),
                round(float(val_loss_all), 4),
                round(float(train_dice), 4),
                round(float(val_dice), 4),
                round(duration, 2),
                round(float(train_metrics['TP']), 4),
                round(float(train_metrics['FP']), 4),
                round(float(train_metrics['TN']), 4),
                round(float(train_metrics['FN']), 4),
                round(float(train_metrics['Precision']), 4),
                round(float(train_metrics['Recall']), 4),
                round(float(val_metrics['TP']), 4),
                round(float(val_metrics['FP']), 4),
                round(float(val_metrics['TN']), 4),
                round(float(val_metrics['FN']), 4),
                round(float(val_metrics['Precision']), 4),
                round(float(val_metrics['Recall']), 4)
            ])

        #save_predictions_as_imgs(
        #    val_loader, model, folder="saved_images/", device=config.DEVICE, num_examples=8, epoche=epoch
        #)

if __name__ == "__main__":
    main()