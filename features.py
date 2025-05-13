# Auto-generated M4‑optimised module – keep lines tight

import numpy as np
from scipy.signal import stft

class FeatureExtractor:
    def __init__(self, fs, n_fft=256, hop=128):
        self.fs, self.n_fft, self.hop = fs,n_fft,hop
    def stft_tensor(self, x):
        if x.ndim==1: x=x[None]
        out=[]
        for seg in x:
            _,_,Z=stft(seg,self.fs,'hann',self.n_fft,self.hop)
            mag=np.abs(Z)[:128,:128]
            img=np.log1p(mag); img=(img-img.min())/(img.ptp()+1e-9)
            out.append(img.astype('float32'))
        return np.stack(out)
