# Auto-generated M4‑optimised module – keep lines tight

import torch, torch.nn as nn, torch.nn.functional as F

def _blk(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,1,1),nn.BatchNorm2d(cout),nn.ReLU(True),
                         nn.Conv2d(cout,cout,5,1,2),nn.BatchNorm2d(cout),nn.ReLU(True),
                         nn.MaxPool2d(2))
class MSCNN(nn.Module):
    def __init__(self,in_channels=1,out_classes=2):
        super().__init__()
        self.b1=_blk(in_channels,32); self.b2=_blk(32,64)
        self.c3=nn.Conv2d(64,128,3,1,1); self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(128,128); self.out=nn.Linear(128,out_classes)
    def forward(self,x):
        x=self.b1(x); x=self.b2(x); x=F.relu(self.c3(x))
        x=self.gap(x).flatten(1); x=F.relu(self.fc(x)); return self.out(x)
