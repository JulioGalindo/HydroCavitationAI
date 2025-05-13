
"""Multi‑Scale Convolutional Neural Network definition."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)
