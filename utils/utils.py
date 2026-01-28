import torch
import os
from utils.dice_score import dice_coeff
from torchvision.utils import make_grid
import torchvision.utils as vutils
from perun import monitor
from utils.dice_score import TopKDiceLoss

@torch.no_grad()
def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    for batch in loader:
            x = batch['image'].to(device)
            true_masks = batch['mask'].to(device)

            if true_masks.dim() == 3:
                true_masks = true_masks.unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == true_masks).sum()
            num_pixels += torch.numel(preds)
            
            dice_score += dice_coeff(preds, true_masks, reduce_batch_first=False)

    print(f"Accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice Score: {dice_score/len(loader)}")
    
    model.train()
    return dice_score / len(loader)


@torch.no_grad()
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cpu", num_examples=8, epoche = 0):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    images_to_save = []
    
    for idx, batch in enumerate(loader):
            if idx > 0: 
                break
            
            imgs = batch['image'].to(device)
            true_masks = batch['mask'].to(device)

            if true_masks.dim() == 3:
                true_masks = true_masks.unsqueeze(1)
            
            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float()

            for i in range(min(num_examples, imgs.shape[0])):
                
                img_single = imgs[i]        # [1, H, W]
                img_disp = img_single * 0.5 + 0.5
                img_disp = torch.clamp(img_disp, 0, 1)
                
                true_single = true_masks[i] # [1, H, W]
                pred_single = preds[i]      # [1, H, W]

                # Convert to RGB for better visualization
                img_rgb = img_disp.repeat(3, 1, 1).float()
                true_rgb = true_single.repeat(3, 1, 1).float()
                pred_rgb = pred_single.repeat(3, 1, 1).float()

                overlay = img_rgb * 0.6 
                
                tp = (pred_single == 1) & (true_single == 1)
                fp = (pred_single == 1) & (true_single == 0)
                fn = (pred_single == 0) & (true_single == 1)

                overlay[1, :, :] += tp.squeeze().float() * 0.4  # Green
                overlay[0, :, :] += fp.squeeze().float() * 0.8  # Red
                overlay[2, :, :] += fn.squeeze().float() * 0.8  # Blue

                overlay = torch.clamp(overlay, 0, 1)
                combined = torch.cat((img_rgb, true_rgb, pred_rgb, overlay), dim=2)
                
                images_to_save.append(combined)

    grid = make_grid(images_to_save, nrow=4, padding=10, pad_value=0.5)
    save_path = os.path.join(folder, "prediction_grid.png")
    vutils.save_image(grid, f"{folder}/prediction_epoch_{epoche}.png")
    
    print(f"Visualization saved: {save_path}")
    model.train()

@monitor()
@torch.no_grad()
def evaluate(loader, model, loss_fn, device, threshold=0.5):
    model.eval()
    loss_fn_all = TopKDiceLoss(k=100)
    running_loss = 0.0
    running_loss_all = 0.0
    total_samples = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0

    TP = FP = FN = TN = 0
    steps = 0

    for batch in loader:
        images = batch['image'].to(device)
        true_masks = batch['mask'].to(device)

        if true_masks.dim() == 3:
            true_masks = true_masks.unsqueeze(1).float()
        else:
            true_masks = true_masks.float()

        true_masks = (true_masks > 0.5).float()
        logits = model(images) 

        loss = loss_fn(logits, true_masks)
        loss_all = loss_fn_all(logits, true_masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_loss_all += loss_all.item() * batch_size
        total_samples += batch_size

        # Probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Accuracy
        num_correct += (preds == true_masks).sum().item()
        num_pixels += torch.numel(preds)
        
        # Confusion
        TP += ((preds == 1) & (true_masks == 1)).sum().item()
        FP += ((preds == 1) & (true_masks == 0)).sum().item()
        FN += ((preds == 0) & (true_masks == 1)).sum().item()
        TN += ((preds == 0) & (true_masks == 0)).sum().item()

        # Dice
        batch_dice = dice_coeff(
            preds,
            true_masks,
            reduce_batch_first=False
        )
        dice_score += batch_dice.item() * batch_size

    avg_loss = running_loss / total_samples
    avg_loss_all = running_loss_all / total_samples

    acc = num_correct / num_pixels * 100 if num_pixels > 0 else 0
    avg_dice = dice_score / total_samples
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)

    return {
        "Loss": avg_loss,
        "Loss_All": avg_loss_all,
        "Accuracy": acc,
        "Dice": avg_dice,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": precision,
        "Recall": recall,
    }