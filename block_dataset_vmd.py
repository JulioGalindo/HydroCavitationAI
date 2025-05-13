
"""BlockDataset with SSA‑optimised VMD preprocessing (M‑series aware)."""
from __future__ import annotations
import json, torch, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from preprocess import Preprocessor
from features import FeatureExtractor
from vmd import VMD

class BlockDatasetVMD(Dataset):
    """Each item → (tensor[M,H,W], label) where tensor comes from selected VMD modes."""
    def __init__(self, root:Path, rpm:float, fs:float, config_path:Path):
        self.sigs = np.load(root/"signals.npy", mmap_mode="r")
        self.labels = np.load(root/"labels.npy")
        cfg=json.load(open(config_path))
        self.alpha,self.K,self.tau = cfg["alpha"], cfg["K"], cfg["tau"]
        blen_s = 4*60/rpm
        self.samples=int(blen_s*fs)
        self.blocks=self.labels.shape[1]
        self.pre=Preprocessor(fs,band=(1e5,4.5e5),seg_length=blen_s)
        self.feat=FeatureExtractor(fs,256,128)
        self.vmd = VMD(self.alpha, self.K, self.tau)   # keep on CPU or MPS

    def __len__(self): return self.sigs.shape[0]*self.blocks
    def __getitem__(self, idx):
        sig_idx, blk = divmod(idx,self.blocks)
        start=blk*self.samples; end=start+self.samples
        chunk=self.sigs[sig_idx,start:end]
        seg=self.pre.normalize(self.pre.bandpass(chunk))
        modes = self.vmd.decompose(seg)
        sel=modes[-3:]  # take highest 3 freq modes
        stacks=[self.feat.stft_tensor(m)[0] for m in sel]
        X=np.stack(stacks,axis=0)
        y=self.labels[sig_idx,blk]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y)
