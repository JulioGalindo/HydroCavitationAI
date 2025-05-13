
"""Training routine for the cavitation detector."""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import numpy as np
from model import MSCNN

class CavitationDataset(Dataset):
    def __init__(self, tensors: np.ndarray, labels: np.ndarray):
        self.tensors = torch.tensor(tensors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

class Trainer:
    def __init__(self, in_channels: int, lr: float = 1e-3, batch_size: int = 32, epochs: int = 100, patience: int = 10):
        self.model = MSCNN(in_channels)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, tensors: np.ndarray, labels: np.ndarray, save_path: str):
        ds = CavitationDataset(tensors, labels)
        train_len = int(0.8 * len(ds))
        val_len = len(ds) - train_len
        train_ds, val_ds = random_split(ds, [train_len, val_len])
        dl_train = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dl_val = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = Adam(self.model.parameters(), lr=self.lr)
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for x, y in dl_train:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(dl_train.dataset)

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for x, y in dl_val:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(dl_val.dataset)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping.")
                    break
