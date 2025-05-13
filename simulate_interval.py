# Auto-generated M4‑optimised module – keep lines tight

import argparse, platform, numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from torch import sin, rand, linspace, pi, randn, fft, device as _dev
_APPLE = platform.system()=='Darwin' and platform.machine().startswith('arm')

def _base(n,fs,seed,device):
    t=linspace(0,n/fs,n,device=device); rand.seed(seed)
    f0=rand(1,device=device)*5e3+5e3
    s=0.8*sin(2*pi*f0*t)
    s+=0.1*sin(4*pi*f0*t)+0.1*sin(6*pi*f0*t)
    s+=0.05*randn(n,device=device)
    return s

def _burst(sig,fs,start,len_):
    t=linspace(0,len_/fs,len_,device=sig.device)
    f=rand(1,device=sig.device)*3e5+2e5
    sig[start:start+len_]+=0.3*sin(2*pi*f*t)

def _cpu_task(args):
    idx,n,spb,fs,dtype=args
    import numpy as np, torch
    sig=_base(n,fs,idx,'cpu')
    nb=n//spb; cav=np.zeros(nb,np.uint8); cav[1::2]=1
    for b in np.where(cav)[0]: _burst(sig,fs,b*spb,spb)
    return sig.numpy().astype(dtype),cav

def generate(out:Path,n:int,dur:float,rpm:float,fs:float,dtype:str='float16'):
    out.mkdir(parents=True,exist_ok=True)
    blen=4*60/rpm; spb=int(blen*fs); nb=int(dur//blen); ns=nb*spb
    print(f'[INFO] {n} sig ×{dur}s  blocks={nb}')
    sig_map=np.lib.format.open_memmap(out/'signals.npy','w+',dtype=dtype,shape=(n,ns))
    lbl=np.lib.format.open_memmap(out/'labels.npy','w+',dtype='uint8',shape=(n,nb))
    if _APPLE:
        import torch
        for i in range(n):
            sig=_base(ns,fs,i,'mps')
            cav=torch.zeros(nb,dtype=torch.bool,device='mps'); cav[1::2]=1
            for b in torch.where(cav)[0]: _burst(sig,fs,b*spb,spb)
            sig_map[i]=sig.cpu().numpy().astype(dtype); lbl[i]=cav.cpu().numpy()
    else:
        with Pool(cpu_count()) as p:
            for i,(s,l) in enumerate(p.imap_unordered(_cpu_task,[(i,ns,spb,fs,dtype) for i in range(n)])):
                sig_map[i]=s; lbl[i]=l
    sig_map.flush();lbl.flush()
