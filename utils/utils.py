import torch
from utils.dice_score import dice_coeff
from perun import monitor
from utils.dice_score import TopKDiceLoss

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

    for batch in loader:
        images = batch['image'].to(device)
        true_masks = batch['mask'].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):

            if true_masks.dim() == 4 and true_masks.shape[1] == 1:
                true_masks = true_masks.squeeze(1) 

            true_masks = (true_masks > 0.5).float()
            logits = model(images) 

            loss = loss_fn(logits, true_masks)
            loss_all = loss_fn_all(logits, true_masks)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_loss_all += loss_all.item() * batch_size
            total_samples += batch_size

            # Probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]
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
                preds.unsqueeze(1),
                true_masks.unsqueeze(1),
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