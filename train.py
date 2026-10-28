"""
Training script for MCAR-Net (and the Table 3 ablation variants).

Training configuration used for the reported results:

  * seed 42 (monai.utils.set_determinism), batch size 1, full 128^3 volumes (no patches)
  * Dice + Cross-Entropy loss: MONAI DiceCELoss(softmax=True) with equal weights
    (lambda_dice = lambda_ce = 1, i.e. lambda = 0.5 in Eq. 15 up to a constant factor).
    --lambda-dice / --lambda-ce are used for the loss ablation (Table 3).
  * AdamW, lr 1e-2, weight decay 1e-5, betas (0.9, 0.999), eps 1e-8
  * ReduceLROnPlateau(factor 0.5, patience 3, min_lr 1e-7) stepped on the validation loss
    with mode='max' (the setting used for all reported models, kept for reproducibility).
    Because the validation loss decreases during training, mode='max' counts every epoch
    as "no improvement", so in practice the learning rate is halved every four epochs
    until it reaches 1e-7, i.e. the scheduler acts as a step decay.
    --scheduler-mode min gives the conventional plateau behaviour.
  * Early stopping: training stops when the validation IoU has not improved for
    25 consecutive epochs (--patience 25), or at the epoch limit (--epochs).
    Epoch limits used: BraTS 2019 = 300, BraTS 2020 = 200, BraTS 2021 = 100.
    Early stopping triggered in all three cases; the best checkpoints were obtained at
    epochs 200, 125 and 35 (training stopped at epochs 225, 150 and 60).
  * Model selection: the checkpoint with the highest validation IoU (best.pth) is used
    for all reported test results (evaluate.py). This IoU is a training-time monitoring
    quantity only (0.5 threshold applied to the raw network outputs, averaged over the
    4 one-hot channels); it is not the evaluation metric.
  * Resuming: last.pth stores the full training state (model, optimizer, scheduler,
    epoch, best validation IoU, early-stopping counter) after every epoch. Training can
    be interrupted and continued with --resume, which is equivalent to an uninterrupted run.

Usage:
    python train.py --data-dir data/brats2020 --out-dir runs/brats2020 --epochs 200
    python train.py --data-dir data/brats2020 --out-dir runs/brats2020 --epochs 200 \
                    --resume runs/brats2020/last.pth
"""

import argparse
import csv
import os

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data import make_loader
from model import ABLATION_VARIANTS, build_model


def monitor_iou(preds, labels):
    """Training-time monitoring IoU over the 4 one-hot channels (checkpoint selection only)."""
    preds, labels = preds > 0.5, labels > 0.5
    inter = torch.logical_and(preds, labels).sum(dim=(2, 3, 4))
    union = torch.logical_or(preds, labels).sum(dim=(2, 3, 4))
    return (inter / (union + 1e-6)).mean().item()


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    post = AsDiscrete(threshold=0.5)
    losses, ious = [], []
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, leave=False):
            x, y = batch["image"].to(device), batch["label"].to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            ious.append(monitor_iou(post(out), post(y)))
    return float(np.mean(losses)), float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with Train/ and Val/ from preprocess.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="mcar_net", choices=list(ABLATION_VARIANTS))
    ap.add_argument("--epochs", type=int, default=200, help="epoch limit")
    ap.add_argument("--patience", type=int, default=25, help="early-stopping patience (epochs)")
    ap.add_argument("--resume", default=None, help="full training state to resume from (last.pth)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lambda-dice", type=float, default=1.0)
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--scheduler-mode", default="max", choices=["max", "min"],
                    help="'max' = setting used for the reported models (see module docstring)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, train_files = make_loader(os.path.join(args.data_dir, "Train"), train=True)
    val_loader, val_files = make_loader(os.path.join(args.data_dir, "Val"))
    print(f"train {len(train_files)}  val {len(val_files)}")

    model = build_model(args.variant).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                      betas=(0.9, 0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(optimizer, mode=args.scheduler_mode, factor=0.5,
                                  patience=3, min_lr=1e-7)
    loss_fn = DiceCELoss(softmax=True, lambda_dice=args.lambda_dice, lambda_ce=args.lambda_ce)

    start_epoch, best_iou, best_epoch, counter = 1, -float("inf"), 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_iou, best_epoch, counter = state["best_iou"], state["best_epoch"], state["counter"]
        print(f"resumed from {args.resume} at epoch {start_epoch} "
              f"(best val IoU {best_iou:.4f} at epoch {best_epoch}, counter {counter})")

    best_file = os.path.join(args.out_dir, "best.pth")
    log_path = os.path.join(args.out_dir, "history.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["epoch", "lr", "train_loss", "train_iou", "val_loss", "val_iou"])

        for ep in range(start_epoch, args.epochs + 1):
            tr_loss, tr_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
            va_loss, va_iou = run_epoch(model, val_loader, loss_fn, device)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(va_loss)

            msg = ""
            if va_iou > best_iou:
                best_iou, best_epoch, counter = va_iou, ep, 0
                torch.save(model.state_dict(), best_file)   # weights only, used by evaluate.py
                msg = "  <- best"
            else:
                counter += 1

            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": ep, "best_iou": best_iou,
                        "best_epoch": best_epoch, "counter": counter},
                       os.path.join(args.out_dir, "last.pth"))
            log.writerow([ep, lr, tr_loss, tr_iou, va_loss, va_iou])
            fh.flush()
            print(f"epoch {ep} | lr {lr:.2e} | train loss {tr_loss:.4f} iou {tr_iou:.4f} "
                  f"| val loss {va_loss:.4f} iou {va_iou:.4f} | no-improve {counter}/{args.patience}{msg}")

            if counter >= args.patience:
                print(f"early stopping at epoch {ep}")
                break

    print(f"done. best val IoU {best_iou:.4f} at epoch {best_epoch} -> {best_file}")


if __name__ == "__main__":
    main()
