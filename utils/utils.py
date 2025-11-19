import torch
import os
from utils.dice_score import dice_coeff
from torchvision.utils import make_grid
import torchvision.utils as vutils

def save_checkpoint(state, filename="checkpoint.pth."):
    print("=> Speichere Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Lade Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
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

    print(f"Genauigkeit: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice Score: {dice_score/len(loader)}")
    
    model.train()
    return dice_score / len(loader)



def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cpu", num_examples=8, epoche = 0):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    
    images_to_save = []
    
    with torch.no_grad():
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
                true_single = true_masks[i] # [1, H, W]
                pred_single = preds[i]      # [1, H, W]

                # In RGB umwandeln zur besseren Visualisierung
                img_rgb = img_single.repeat(3, 1, 1)
                true_rgb = true_single.repeat(3, 1, 1)
                pred_rgb = pred_single.repeat(3, 1, 1)

                # Overlay erstellen
                overlay = img_rgb * 0.6 
                
                tp = (pred_single == 1) & (true_single == 1)
                fp = (pred_single == 1) & (true_single == 0)
                fn = (pred_single == 0) & (true_single == 1)

                overlay[1, :, :] += tp.squeeze() * 0.4  # Grün
                overlay[0, :, :] += fp.squeeze() * 0.8  # Rot
                overlay[2, :, :] += fn.squeeze() * 0.8  # Blau

                overlay = torch.clamp(overlay, 0, 1)
                combined = torch.cat((img_rgb, true_rgb, pred_rgb, overlay), dim=2)
                
                images_to_save.append(combined)

    grid = make_grid(images_to_save, nrow=4, padding=10, pad_value=0.5)
    save_path = os.path.join(folder, "prediction_grid.png")
    vutils.save_image(grid, f"{folder}/prediction_epoch_{epoche}.png")
    
    print(f"Visualisierung gespeichert: {save_path}")
    model.train()