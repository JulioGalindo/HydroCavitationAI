
"""Feature extraction utilities: STFT and tensor conversion."""
import numpy as np
from scipy.signal import stft

class FeatureExtractor:
    def __init__(self, fs: float, n_fft: int = 256, hop: int = 128):
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop

    def stft_tensor(self, segments: np.ndarray):
        """Compute log‑magnitude STFT tensor for each segment.

        Parameters
        ----------
        segments : np.ndarray
            Shape (N, L) where L is samples per segment.

        Returns
        -------
        np.ndarray
            Tensor with shape (N, n_freq, n_time)
        """
        tensors = []
        for seg in segments:
            _, _, Zxx = stft(seg, fs=self.fs, window='hamming',
                             nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
            S = np.log1p(np.abs(Zxx))
            tensors.append(S)
        return np.stack(tensors)
