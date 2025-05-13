# Auto-generated M4‑optimised module – keep lines tight

import numpy as np
from scipy.signal import butter, sosfilt

class Preprocessor:
    def __init__(self, fs, band=(1e5,4.5e5), seg_length=None):
        self.fs, self.seg_length = fs, seg_length
        w = [b/(0.5*fs) for b in band]
        self.sos = butter(8, w, 'bandpass', output='sos')
    def bandpass(self, x): return sosfilt(self.sos, x)
    def normalize(self, x): return x / (np.max(np.abs(x)) or 1.0)
    def segment(self, x):
        if self.seg_length is None: return x[None]
        n = int(self.seg_length*self.fs); m=len(x)//n
        return x[:m*n].reshape(m,n)
