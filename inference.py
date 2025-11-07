import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import yaml
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt
import warnings

# Disable Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore")

from dataset import ISIC_2017_Seg_Test_Dataset
from util import seed_everything
from models import ISIC_2017_Seg_Model

# Define DiceLoss and TverskyLoss (kept here even though not used in testing)
DiceLoss = smp.losses.DiceLoss(mode="binary")
TverskyLoss = smp.losses.TverskyLoss(mode="binary", log_loss=False)


# ---------------------- metric helpers ----------------------
def compute_iou(preds, targets, num_classes=2):
    """Compute mean IoU over classes for 0/1 masks"""
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = intersection / union if union != 0 else torch.tensor(1.0, device=preds.device)
        iou_per_class.append(iou)
    return torch.mean(torch.stack(iou_per_class))


def compute_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return correct / total


def compute_prf1(preds_bin, targets, eps=1e-7):
    """
    Compute Precision, Recall, F1 for binary (0/1) tensors.
    F1 is equivalent to Dice here.
    """
    preds_bin = preds_bin.float()
    targets = targets.float()
    TP = (preds_bin * targets).sum()
    FP = (preds_bin * (1 - targets)).sum()
    FN = ((1 - preds_bin) * targets).sum()
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision.item(), recall.item(), f1.item()


# ---------------------- visualization / failure dump ----------------------
def save_failure_figure(orig_path, gt_mask, pred_mask, save_path):
    """
    Save a 3-panel figure: image / GT / prediction.
    gt_mask / pred_mask can be numpy uint8 (0/255) or 0/1.
    """
    try:
        img = Image.open(orig_path).convert("RGB")
    except Exception:
        # If original image cannot be loaded, fall back to a dummy image (very rare)
        img = Image.fromarray((np.stack([pred_mask] * 3, axis=-1) * 255).astype(np.uint8))

    gt_vis = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1.0 else gt_mask.astype(np.uint8)
    pr_vis = (pred_mask * 255).astype(np.uint8) if pred_mask.max() <= 1.0 else pred_mask.astype(np.uint8)

    plt.figure(figsize=(10, 3.2))

    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Image")

    # ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(gt_vis, cmap="gray")
    plt.axis("off")
    plt.title("Ground Truth")

    # prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pr_vis, cmap="gray")
    plt.axis("off")
    plt.title("Prediction")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------- load trained model ----------------------
def load_best_model(model, pth_path):
    if os.path.exists(pth_path):
        model.load_state_dict(torch.load(pth_path, map_location=torch.device("cuda")))
        print(f"[OK] Loaded model from: {pth_path}")
    else:
        print(f"[WARN] Cannot find model at: {pth_path}")
    return model


# ---------------------- main ----------------------
parser = argparse.ArgumentParser(description="Test ISIC 2017 segmentation model")
parser.add_argument("--cfg", default="configs/mit_b5_with_glr_fpn.yaml", type=str)
parser.add_argument("--test_csv", default="./datasets/test.csv", type=str,
                    help="Path to the ISIC 2017 test CSV")
parser.add_argument("--pth", default="./weights/2017_weights.pth", type=str,
                    help="Path to the model weights (.pth)")

# failure case output
parser.add_argument("--fail_dir", default="./fails_isic2017", type=str,
                    help="Directory to save failure cases")
parser.add_argument("--fail_topk", default=12, type=int,
                    help="Save the worst-K samples by Dice/F1")
parser.add_argument("--fail_thresh", default=None, type=float,
                    help="If set, also save all samples with Dice below this threshold (0~1). "
                         "Will be merged with top-k list.")

# result CSV
parser.add_argument("--out_csv", default="./GLR_pth/GLR_Net_2017_results.csv", type=str)

args = parser.parse_args()
seed_everything(seed=123)

if __name__ == "__main__":
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # dataset
    test_df = pd.read_csv(args.test_csv)
    test_dataset = ISIC_2017_Seg_Test_Dataset(df=test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg["workers"],
        shuffle=False
    )

    # model
    model = ISIC_2017_Seg_Model(
        encoder_name=cfg["encoder_name"],
        encoder_weights=cfg["encoder_weights"],
        decoder_name=cfg["decoder_name"]
    ).cuda()
    model = load_best_model(model, args.pth)

    print("\n[INFO] Start testing ISIC 2017 ...")
    model.eval()

    rows = []
    total_dice = total_iou = total_acc = total_f1 = 0.0
    total_prec = total_rec = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, masks, image_path in tqdm(test_loader, desc="Testing ISIC 2017"):
            images, masks = images.cuda(), masks.cuda()
            preds = model(images)
            preds_prob = torch.sigmoid(preds)
            preds_bin = (preds_prob > 0.5).float()  # (B,1,H,W) in {0,1}

            # Dice (equivalent to F1 here)
            inter = (preds_bin * masks).sum()
            union = preds_bin.sum() + masks.sum()
            dice = (2.0 * inter) / (union + 1e-7)

            # IoU, Acc, Precision, Recall, F1
            iou = compute_iou(preds_bin, masks)
            acc = compute_accuracy(preds_bin, masks)
            prec, rec, f1 = compute_prf1(preds_bin, masks)

            dice_val = dice.item()
            iou_val = iou.item()
            acc_val = acc.item()

            total_dice += dice_val
            total_iou += iou_val
            total_acc += acc_val
            total_prec += prec
            total_rec += rec
            total_f1 += f1
            num_samples += 1

            rows.append(
                {
                    "image_path": image_path[0],
                    "dice": dice_val,
                    "iou": iou_val,
                    "accuracy": acc_val,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )

    # averages
    avg_dice = total_dice / num_samples
    avg_iou = total_iou / num_samples
    avg_acc = total_acc / num_samples
    avg_prec = total_prec / num_samples
    avg_rec = total_rec / num_samples
    avg_f1 = total_f1 / num_samples

    print("\n[RESULT] ISIC 2017 overall test metrics:")
    print(f"  Dice (F1): {avg_dice:.4f}  (F1={avg_f1:.4f})")
    print(f"  IoU:       {avg_iou:.4f}")
    print(f"  Accuracy:  {avg_acc:.4f}")
    print(f"  Precision: {avg_prec:.4f}")
    print(f"  Recall:    {avg_rec:.4f}")

    # save CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Testing finished ({num_samples} images). Results saved to {args.out_csv}")

    # ----------------- dump failure cases -----------------
    fail_indices = set()

    # worst-K by Dice
    if args.fail_topk and args.fail_topk > 0:
        worst_k = df.sort_values("dice", ascending=True).head(args.fail_topk).index.tolist()
        fail_indices.update(worst_k)

    # below-threshold
    if args.fail_thresh is not None:
        under_t = df[df["dice"] < float(args.fail_thresh)].index.tolist()
        fail_indices.update(under_t)

    fail_indices = sorted(list(fail_indices))

    if len(fail_indices) > 0:
        os.makedirs(args.fail_dir, exist_ok=True)
        print(f"[INFO] Dumping failure cases ({len(fail_indices)} samples) to: {args.fail_dir}")

        cache = {r["image_path"]: r for _, r in df.iterrows()}

        model.eval()
        with torch.no_grad():
            for images, masks, image_path in DataLoader(
                test_dataset, batch_size=1, num_workers=0, shuffle=False
            ):
                ipath = image_path[0]
                if ipath not in cache:
                    continue
                idx = df.index[df["image_path"] == ipath][0]
                if idx not in fail_indices:
                    continue

                images = images.cuda()
                preds = model(images)
                preds_bin = (torch.sigmoid(preds) > 0.5).float().cpu().numpy()[0, 0]  # (H,W)
                gt_mask = masks.numpy()[0, 0]  # (H,W)

                base = os.path.splitext(os.path.basename(ipath))[0]
                out_path = os.path.join(
                    args.fail_dir,
                    f"{idx:04d}_{base}_dice{cache[ipath]['dice']:.3f}.png",
                )
                save_failure_figure(ipath, gt_mask, preds_bin, out_path)

        print("[OK] Failure case export finished.")
    else:
        print("[INFO] No failure cases selected. Adjust --fail_topk or --fail_thresh if needed.")
