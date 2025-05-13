
"""Per‑block cavitation probability timeline analyzer."""
from __future__ import annotations
import argparse, platform, csv, numpy as np, torch, soundfile as sf
from pathlib import Path
from preprocess import Preprocessor
from features import FeatureExtractor
from model import MSCNN

_IS_APPLE=platform.system()=="Darwin" and platform.machine().startswith("arm")
_HAS_MPS=torch.backends.mps.is_available()

def load_raw(p:Path):
    if p.suffix==".wav": x,fs=sf.read(p); x=x.mean(1) if x.ndim>1 else x; return x,fs
    if p.suffix==".npy": return np.load(p,mmap_mode="r"),1_000_000
    raise ValueError(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--signal",required=True)
    ap.add_argument("--labels")
    ap.add_argument("--rpm",type=float,default=300)
    ap.add_argument("--fs",type=float,default=1_000_000)
    ap.add_argument("--model",default="weights_blocks.pth")
    ap.add_argument("--csv")
    args=ap.parse_args()
    x,fs=load_raw(Path(args.signal))
    if fs!=args.fs: raise SystemExit("resample first")
    block_sec=4*60/args.rpm
    spb=int(block_sec*fs); n_blocks=len(x)//spb; x=x[:n_blocks*spb]
    dev="mps" if _IS_APPLE and _HAS_MPS else "cpu"
    pre=Preprocessor(fs,band=(1e5,4.5e5),seg_length=block_sec)
    feat=FeatureExtractor(fs,256,128)
    mdl=MSCNN(1).to(dev); mdl.load_state_dict(torch.load(args.model, map_location=dev)); mdl.eval()
    probs=[]
    for b in range(n_blocks):
        chunk=x[b*spb:(b+1)*spb]
        seg=pre.segment(pre.normalize(pre.bandpass(chunk)))
        X=feat.stft_tensor(seg)[:,None,...]
        with torch.no_grad():
            with torch.autocast(device_type=dev,dtype=torch.float16) if dev!="cpu" else torch.no_grad():
                p=torch.softmax(mdl(torch.tensor(X,device=dev)),1)[:,1].mean().item()
        probs.append(p)
        print(f"{b:03d}: prob={p:.3f} {'cav' if p>=0.5 else 'no-cav'}")
    if args.csv:
        with open(args.csv,"w",newline="") as f: w=csv.writer(f); w.writerow(["block","prob"]); w.writerows(enumerate(probs))

if __name__=="__main__": main()
