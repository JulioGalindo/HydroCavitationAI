
"""Block-wise trainer for MSCNN (per 4‑rev window).

• Auto‑device: MPS on Apple Silicon or CPU.
• Mixed‑precision (`torch.autocast`) for float16 on GPU.
"""
from __future__ import annotations
import argparse, platform, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn, torch.optim as optim
from tqdm import trange
from preprocess import Preprocessor
from features import FeatureExtractor
from model import MSCNN

_IS_APPLE = platform.system()=="Darwin" and platform.machine().startswith("arm")
_HAS_MPS  = torch.backends.mps.is_available()

class BlockDS(Dataset):
    def __init__(self, root:Path, rpm:float, fs:float):
        self.sigs   = np.load(root/"signals.npy", mmap_mode="r")
        self.labels = np.load(root/"labels.npy")
        self.fs=fs
        blen_s = 4*60/rpm
        self.samples = int(blen_s*fs)
        self.blocks = self.labels.shape[1]
        self.pre  = Preprocessor(fs, band=(1e5,4.5e5), seg_length=blen_s)
        self.feat = FeatureExtractor(fs, 256, 128)
    def __len__(self): return self.sigs.shape[0]*self.blocks
    def __getitem__(self, idx):
        sig_idx, blk = divmod(idx, self.blocks)
        start=blk*self.samples; end=start+self.samples
        chunk=self.sigs[sig_idx,start:end]
        seg=self.pre.segment(self.pre.normalize(self.pre.bandpass(chunk)))
        X=self.feat.stft_tensor(seg)[0][None]
        y=self.labels[sig_idx, blk]
        return torch.tensor(X), torch.tensor(y)

def train(root:Path, rpm:float, fs:float, epochs=30, batch=64, lr=1e-3,
          patience=6, device="cpu"):
    ds=BlockDS(root,rpm,fs)
    vs=int(0.2*len(ds)); ts=len(ds)-vs
    tr_ds,va_ds=random_split(ds,[ts,vs],generator=torch.Generator().manual_seed(42))
    tr=DataLoader(tr_ds,batch,shuffle=True)
    va=DataLoader(va_ds,batch)
    model=MSCNN(1).to(device)
    opt=optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    scaler=None
    use_amp = device!="cpu"
    best=float("inf"); patience_left=patience
    for epoch in trange(epochs,desc="Epoch"):
        model.train(); tot=n=0
        for X,y in tr:
            X,y=X.to(device),y.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.autocast(device_type=device, dtype=torch.float16):
                    out=model(X)
                    loss=loss_fn(out,y)
                loss.backward()
            else:
                loss=loss_fn(model(X),y); loss.backward()
            opt.step(); tot+=loss.item()*y.size(0); n+=y.size(0)
        tr_loss=tot/n
        model.eval(); tot=n=0
        with torch.no_grad():
            for X,y in va:
                X,y=X.to(device),y.to(device)
                if use_amp:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        loss=loss_fn(model(X),y)
                else:
                    loss=loss_fn(model(X),y)
                tot+=loss.item()*y.size(0); n+=y.size(0)
        va_loss=tot/n
        print(f"Epoch {epoch+1}: train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss < best-1e-4:
            best=va_loss; patience_left=patience
            torch.save(model.state_dict(),"weights_blocks.pth")
        else:
            patience_left-=1
            if patience_left==0: break

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="data/mixed")
    ap.add_argument("--rpm",type=float,default=300)
    ap.add_argument("--fs",type=float,default=1_000_000)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args()
    dev="cpu" if args.cpu else ("mps" if _IS_APPLE and _HAS_MPS else "cpu")
    train(Path(args.root),args.rpm,args.fs,args.epochs,args.batch,args.lr,device=dev)
