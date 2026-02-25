import torch
import yaml
import csv
import os
import time
import torch.optim as optim
from torch.profiler import ProfilerActivity
from utils.dice_score import TopKDiceLoss
import argparse
from unet.unet_model import UNet
from utils.data_loading import BasicDataset, get_dataloader
from utils.utils import evaluate
from perun import monitor
torch.set_float32_matmul_precision('high')

@monitor()
def train_fn(loader, model, optimizer, loss_fn, device, profiler=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        # Load data to device
        data_img = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True)

        #Forward Pass
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()
        
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16): 
            predictions = model(data_img)
            loss = loss_fn(predictions.float(), targets)

        # Backward Pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()    

        current_batch_size = data_img.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size
        
        if profiler:
            profiler.step()

    return running_loss / total_samples

@monitor()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Hyperparameter
    learning_rate = config['hyperparameters']['learning_rate']
    batch_size = config['hyperparameters']['batch_size']
    num_epoch = config['hyperparameters']['num_epochs']
    target_size = tuple(config['hyperparameters']['target_size'])
    kernel_flex = config['hyperparameters']['kernel_flex']
    if kernel_flex:
        kernel_str = "flex"
    else:        
        kernel_str = "fixed"
    
    # Model
    channels_in = config['model']['in_channels']
    num_channels = config['model']['num_classes']
    
    # Paths
    base_dir = config['paths']['base_dir']
    log_dir = config['paths']['log_dir']
    res_str = str(target_size[1])
    train_img_dir = os.path.join(base_dir, res_str, "train", "image")
    train_mask_dir = os.path.join(base_dir, res_str, "train", "mask")
    val_img_dir = os.path.join(base_dir, res_str, "val", "image")
    val_mask_dir = os.path.join(base_dir, res_str, "val", "mask")
    seed = config['hyperparameters']['seed']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet(
        n_channels=channels_in,
        n_classes=num_channels, 
        input_res=target_size[1], 
        kernel_flex=kernel_flex
    ).to(device)
    loss_fn = TopKDiceLoss(k=50, smooth=1e-5) 

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4, 
        betas = (0.9, 0.999),
        eps=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        patience=4,    
        factor=0.8,   
        threshold=1e-3,
        min_lr=1e-6
    )

    train_ds = BasicDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        is_train=True
    )

    train_loader = get_dataloader(
        config_dict=config,
        dataset= train_ds,
        mode="train",
        batch_size= batch_size
    )

    val_ds = BasicDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        is_train=False
    )

    val_loader = get_dataloader(
        config_dict=config,
        dataset = val_ds,
        mode="val",
        batch_size= batch_size
    )

    os.makedirs(log_dir, exist_ok=True)
    log_file = f"log_{res_str}_{num_epoch}_{learning_rate}_{kernel_str}_{seed}_unet.csv"
    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                'Num_Params',
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
                'Peak_VRAM_GB', 
                'FLOPs_per_image',
                'Latency_ms_per_image',
                'Throughput_img_per_sec',
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
    
    trace_dir = os.path.join(
    log_dir, 
    f"trace_{res_str}_{num_epoch}_{learning_rate}_{kernel_str}_{seed}")

    os.makedirs(trace_dir, exist_ok=True)

    # Calculate the number of batches per epoch to determine the profiling schedule
    batches_per_epoch = len(train_loader)
    wait_steps = max(0, batches_per_epoch - 6)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait_steps, warmup=3, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        with_flops=True,
        record_shapes=True,
        with_modules=True,
        profile_memory=True,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:  
        
        for epoch in range(num_epoch):
            torch.cuda.reset_peak_memory_stats() 
            torch.cuda.synchronize()
            start_time = time.time()


            train_loss = train_fn(loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn,
                                   device=device, profiler=prof)

            val_metrics = evaluate(val_loader, model, loss_fn, device=device)
            train_metrics = evaluate(train_loader, model, loss_fn, device=device)

            scheduler.step(val_metrics['Dice'])

            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time

            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
            num_samples = len(train_loader.dataset)

            latency_ms = (duration / num_samples) * 1000
            throughput = num_samples / duration

            if epoch <= 2:
                raw_flops = sum(item.flops for item in prof.key_averages())
                total_flops = raw_flops / (3 * batch_size)
            else:
                total_flops = 0

            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    sum(p.numel() for p in model.parameters()),
                    int(epoch),
                    int(res_str),
                    optimizer.param_groups[0]["lr"],
                    batch_size,
                    num_samples,
                    len(val_loader.dataset),
                    round(float(train_loss), 4),
                    round(float(val_metrics["Loss"]), 4),
                    round(float(val_metrics["Loss_All"]), 4),
                    round(float(train_metrics["Dice"]), 4),
                    round(float(val_metrics["Dice"]), 4),
                    round(duration, 2),
                    round(peak_vram, 3),
                    total_flops,
                    round(latency_ms, 3),
                    round(throughput, 2),
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

if __name__ == "__main__":
    main()