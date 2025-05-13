# train.py
"""Training script for HydroCavitationAI – MSCNN model
=======================================================
This module handles the **full training workflow**:

* Load `tensors_train.npz` and `tensors_test.npz` (or any custom split)
* Training loop with metric logging (train / validation loss)
* **Graceful Ctrl‑C**: saves a checkpoint + plots before exiting
* `--resume ckpt.pth` to continue a previous run
* Early stopping driven by `--patience`
* Generates two PNGs in the *reports/* folder:
    • **loss_curve.png** – evolution of train / val loss
    • **confusion_matrix.png** – confusion matrix on the validation set

Quick start
-----------
```bash
python train.py \
    --data data/demo/tensors_train.npz \
    --val  data/demo/tensors_test.npz  \
    --epochs 50 --batch 32 --lr 1e-3   \
    --model weights_final.pth          \
    --report
```
Resume an interrupted session:
```bash
python train.py --resume weights_final.pth --data ... --val ... --report
```
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

###############################################################################
# Helper functions – loading & plotting
###############################################################################

def load_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a .npz file and return (X, y) as PyTorch tensors."""
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    return X, y


def plot_loss(train_losses: List[float], val_losses: List[float], out: Path) -> None:
    """Save train / val loss curves as PNG."""
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, out: Path) -> None:
    """Save confusion‑matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(out, dpi=150)
    plt.close()

###############################################################################
# Trainer class encapsulates the entire training workflow
###############################################################################

class Trainer:
    """Self‑contained trainer for the MSCNN cavitation classifier."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Select device – default to CUDA if available unless --cpu flag
        self.device = (torch.device("cuda") if torch.cuda.is_available() and not args.cpu
                        else torch.device("cpu"))

        # ------------------------------------------------------------------
        # 1. Load datasets – training and validation
        # ------------------------------------------------------------------
        X_train, y_train = load_npz(Path(args.data))
        X_val, y_val = load_npz(Path(args.val)) if args.val else (None, None)

        self.train_loader = DataLoader(TensorDataset(X_train, y_train),
                                       batch_size=args.batch, shuffle=True)
        self.val_loader = (DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch)
                           if X_val is not None else None)

        # ------------------------------------------------------------------
        # 2. Create the MSCNN model
        # ------------------------------------------------------------------
        from model import MSCNN  # local import avoids circular dependency
        self.model = MSCNN(num_classes=2).to(self.device)

        # ------------------------------------------------------------------
        # 3. Optionally resume from checkpoint
        # ------------------------------------------------------------------
        if args.resume and Path(args.resume).exists():
            ckpt = torch.load(args.resume, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.start_epoch = ckpt["epoch"] + 1
            self.train_losses = ckpt["train_losses"]
            self.val_losses = ckpt["val_losses"]
            print(f"Resumed from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            self.train_losses: List[float] = []
            self.val_losses: List[float] = []

        # ------------------------------------------------------------------
        # 4. Loss function & optimiser
        # ------------------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Early‑stopping bookkeeping
        self.best_val = float("inf")
        self.patience_counter = 0

        # Folder for plots & checkpoints
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        """Run one epoch; returns average loss."""
        self.model.train(training)
        total, count = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            if training:
                self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            if training:
                loss.backward()
                self.optimizer.step()
            total += loss.item() * yb.size(0)
            count += yb.size(0)
        return total / count

    # ------------------------------------------------------------------
    def _eval_confusion(self):
        """Compute confusion matrix on the validation set."""
        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for xb, yb in self.val_loader:
                preds = self.model(xb.to(self.device)).argmax(dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return y_true, y_pred

    # ------------------------------------------------------------------
    def _save_ckpt(self, epoch: int):
        ckpt = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        torch.save(ckpt, self.args.model)
        print(f"[Checkpoint] epoch {epoch} saved → {self.args.model}")

    # ------------------------------------------------------------------
    def train(self):
        """Main loop with early stopping + graceful Ctrl‑C."""
        try:
            for epoch in trange(self.start_epoch, self.args.epochs, desc="Epoch"):
                tr_loss = self._run_epoch(self.train_loader, training=True)
                self.train_losses.append(tr_loss)

                if self.val_loader is not None:
                    vl_loss = self._run_epoch(self.val_loader, training=False)
                    self.val_losses.append(vl_loss)
                else:
                    vl_loss = 0.0

                print(f"Epoch {epoch+1}: train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f}")

                # --- early stopping logic ---
                if vl_loss < self.best_val - 1e-5:
                    self.best_val = vl_loss
                    self.patience_counter = 0
                    self._save_ckpt(epoch)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.args.patience:
                        print("Early stopping triggered.")
                        break

            # Training finished → generate plots
            if self.args.report and self.val_loader is not None:
                plot_loss(self.train_losses, self.val_losses, self.report_dir / "loss_curve.png")
                y_true, y_pred = self._eval_confusion()
                plot_confusion(y_true, y_pred, self.report_dir / "confusion_matrix.png")
        except KeyboardInterrupt:
            # Ctrl‑C → save checkpoint & plots then exit
            print("\n[Ctrl+C] Interrupted — saving checkpoint and plots…")
            self._save_ckpt(epoch)
            if self.args.report and self.val_loader is not None:
                plot_loss(self.train_losses, self.val_losses, self.report_dir / "loss_curve.png")
                y_true, y_pred = self._eval_confusion()
                plot_confusion(y_true, y_pred, self.report_dir / "confusion_matrix.png")
            print("Checkpoint & plots saved. Resume later with --resume")

###############################################################################
# CLI helper
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train MSCNN cavitation model")
    parser.add_argument("--data", required=True, help="train tensors .npz")
    parser.add_argument("--val", required=True, help="validation tensors .npz")
    parser.add_argument("--model", default="weights_final.pth")
    parser.add_argument("--epochs", type=int, default
