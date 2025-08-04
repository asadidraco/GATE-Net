import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from new import SkinLesionSegmentationModel, SkinLesionDataset  # Update as per your structure

def calculate_dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def calculate_jaccard_index(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt.bool()).sum().item()
    return tp / (tp + fp + 1e-6)

def recall(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    tp = torch.logical_and(pred, gt).sum().item()
    fn = torch.logical_and(~pred.bool(), gt).sum().item()
    return tp / (tp + fn + 1e-6)

def specificity(pred, gt):
    pred = pred > 0.5
    gt = gt > 0.5
    tn = torch.logical_and(~pred.bool(), ~gt.bool()).sum().item()
    fp = torch.logical_and(pred, ~gt.bool()).sum().item()
    return tn / (tn + fp + 1e-6)

def calculate_accuracy(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).sum().float()
    total = target.numel()
    return (correct + smooth) / (total + smooth)

def visualize_results(image, gt, pred, filename, save_dir="./trial/2017_new_test_results/visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    gt_np = gt.cpu().squeeze().numpy()
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    pred_np = torch.sigmoid(pred).cpu().squeeze().numpy()
    axes[2].imshow(pred_np > 0.5, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.png", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SkinLesionSegmentationModel().to(device)
    model.load_state_dict(torch.load('./latest_old/2017_best_model.pth', map_location=device))
    model.eval()

    test_dataset = SkinLesionDataset(
        data_dir='../model/data 2017/test',
        img_size=(256, 256),
        # augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    os.makedirs("./trial/2017_new_test_results/images", exist_ok=True)
    os.makedirs("./trial/2017_new_test_results/gt", exist_ok=True)
    os.makedirs("./trial/2017_new_test_results/predictions", exist_ok=True)
    os.makedirs("./trial/2017_new_test_results/visualizations", exist_ok=True)

    test_metrics = {
        'dice': [],
        'jaccard': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': []
    }

    with torch.no_grad():
        for idx, (image, gt, filename) in enumerate(tqdm(test_loader, desc="Testing")): 
            image, gt = image.to(device), gt.to(device)
            pred = model(image)

            dice_score = calculate_dice_score(pred, gt)
            jaccard_index = calculate_jaccard_index(pred, gt)
            acc = calculate_accuracy(pred, gt)
            prec = precision(pred, gt)
            rec = recall(pred, gt)
            spec = specificity(pred, gt)

            test_metrics['dice'].append(dice_score.item())
            test_metrics['jaccard'].append(jaccard_index.item())
            test_metrics['accuracy'].append(acc.item())
            test_metrics['precision'].append(prec)
            test_metrics['recall'].append(rec)
            test_metrics['specificity'].append(spec)

            visualize_results(image, gt, pred, filename)

            image_np = (image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_np = (gt.cpu().squeeze().numpy() * 255).astype(np.uint8)
            pred_np = (torch.sigmoid(pred).cpu().squeeze().numpy() * 255).astype(np.uint8)

            Image.fromarray(image_np).save(f"./trial/2017_new_test_results/images/{filename}.png")
            Image.fromarray(gt_np).save(f"./trial/2017_new_test_results/gt/{filename}_segmentation.png")
            Image.fromarray((pred_np > 127).astype(np.uint8) * 255).save(f"./trial/2017_new_test_results/predictions/{filename}_pred.png")

    print("\nTest Results:")
    print(f"Dice Score: {np.mean(test_metrics['dice']):.4f} ± {np.std(test_metrics['dice']):.4f}")
    print(f"Jaccard Index: {np.mean(test_metrics['jaccard']):.4f} ± {np.std(test_metrics['jaccard']):.4f}")
    print(f"Accuracy: {np.mean(test_metrics['accuracy']):.4f} ± {np.std(test_metrics['accuracy']):.4f}")
    print(f"Precision: {np.mean(test_metrics['precision']):.4f} ± {np.std(test_metrics['precision']):.4f}")
    print(f"Recall (Sensitivity): {np.mean(test_metrics['recall']):.4f} ± {np.std(test_metrics['recall']):.4f}")
    print(f"Specificity: {np.mean(test_metrics['specificity']):.4f} ± {np.std(test_metrics['specificity']):.4f}")

    with open("./trial/2017_new_test_results/test_metrics.txt", "w") as f:
        f.write("Metric\tMean\tStd\n")
        for metric in test_metrics:
            f.write(f"{metric}\t{np.mean(test_metrics[metric]):.4f}\t{np.std(test_metrics[metric]):.4f}\n")

if __name__ == "__main__":
    main()
