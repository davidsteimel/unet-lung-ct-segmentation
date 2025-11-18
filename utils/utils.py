import torch
import os
import torchvision
from data_loading import BasicDataset
from torch.utils.data import DataLoader
from utils.dice_score import dice_coeff

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
    model.eval() # Modus wechseln: Kein Training!

    with torch.no_grad(): # Keine Gradienten berechnen (spart Speicher)
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            # Dice Score Formel: (2 * Schnittmenge) / (Summe der Elemente)
            dice_score += dice_coeff(preds, y, reduce_batch_first=False)

    print(f"Genauigkeit: {num_correct/num_pixels*100:.2f}%")
    print(f"Dice Score: {dice_score/len(loader)}")
    
    model.train() # Zurück in Trainings-Modus
    return dice_score / len(loader)

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cpu"):

    model.eval()
    os.makedirs(folder, exist_ok = True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        
        break 

    model.train()

import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask, prediction=None):
    
    # CPU und Numpy Konvertierung falls nötig
    if hasattr(img, 'cpu'): img = img.cpu().numpy()
    if hasattr(mask, 'cpu'): mask = mask.cpu().numpy()
    if prediction is not None and hasattr(prediction, 'cpu'): 
        prediction = prediction.cpu().numpy()
    
    # Dimensionen bereinigen 
    if img.ndim == 3: img = img.squeeze(0)
    if mask.ndim == 3: mask = mask.squeeze(0)
    if prediction is not None and prediction.ndim == 3: 
        prediction = prediction.squeeze(0)

    num_plots = 2 if prediction is None else 3
    fig, ax = plt.subplots(1, num_plots, figsize=(10, 5))
    
    # 1. Originalbild
    ax[0].set_title('Input Image')
    ax[0].imshow(img, cmap='gray')
    ax[0].axis("off")

    # 2. Echte Maske
    ax[1].set_title('True Mask')
    ax[1].imshow(mask, cmap='gray')
    ax[1].axis("off")

    # 3. Vorhersage
    if prediction is not None:
        ax[2].set_title('Prediction')
        ax[2].imshow(prediction, cmap='gray')
        ax[2].axis("off")
    
    plt.show()