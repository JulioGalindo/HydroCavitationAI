
"""Predictor with VMD front‑end using SSA‑optimised params."""
from __future__ import annotations
import argparse, json, numpy as np, torch
from pathlib import Path
from preprocess import Preprocessor
from features import FeatureExtractor
from vmd import VMD
from model import MSCNN
import soundfile as sf

def load_raw(p:Path):
    if p.suffix==".wav": x,fs=sf.read(p); x=x.mean(1) if x.ndim>1 else x; return x,fs
    if p.suffix==".npy": return np.load(p),1_000_000
    raise ValueError(p)

class PredictorVMD:
    def __init__(self, model_path:Path, cfg_path:Path, fs:float, device:str):
        self.cfg=json.load(open(cfg_path))
        self.pre=Preprocessor(fs,band=(1e5,4.5e5),seg_length=None)
        self.feat=FeatureExtractor(fs,256,128)
        self.mdl=MSCNN(3).to(device)
        self.mdl.load_state_dict(torch.load(model_path,map_location=device))
        self.mdl.eval(); self.device=device
        self.vmd = VMD(self.cfg["alpha"], self.cfg["K"], self.cfg["tau"])
    def prob(self,chunk:np.ndarray)->float:
        modes = self.vmd.decompose(chunk)
        sel=modes[-3:]
        X=np.stack([self.feat.stft_tensor(m)[0] for m in sel],axis=0)[None]
        with torch.no_grad():
            p=torch.softmax(self.mdl(torch.tensor(X,device=self.device)),1)[0,1].item()
        return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--model",default="weights_blocks_vmd.pth")
    ap.add_argument("--cfg",default="ssa_config.json")
    ap.add_argument("--fs",type=float,default=1_000_000)
    ap.add_argument("--cpu",action="store_true")
    args=ap.parse_args()
    dev="cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    pred=PredictorVMD(Path(args.model),Path(args.cfg),args.fs,dev)
    x,fs=load_raw(Path(args.input))
    print("prob=",pred.prob(x))
if __name__=='__main__': main()
