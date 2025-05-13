
"""Inference utilities for cavitation detection."""
import torch
import numpy as np
from preprocess import Preprocessor
from vmd import VMD
from features import FeatureExtractor
from model import MSCNN

class Predictor:
    def __init__(self, fs: float, vmd_params: dict, model_path: str):
        self.pre = Preprocessor(fs)
        self.vmd = VMD(**vmd_params)
        self.feat = FeatureExtractor(fs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # in_channels will be set after first pass
        self.model = None
        self.model_path = model_path

    def _load_model(self, in_channels: int):
        self.model = MSCNN(in_channels)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, signal: np.ndarray):
        signal_bp = self.pre.bandpass(signal)
        signal_norm = self.pre.normalize(signal_bp)
        segments = self.pre.segment(signal_norm)
        modes = [self.vmd.decompose(seg)[0] for seg in segments]  # take first mode per segment
        modes = np.array(modes)
        tensors = self.feat.stft_tensor(modes)
        tensors = tensors[:, None, ...]  # Add channel dimension

        if self.model is None:
            self._load_model(tensors.shape[1])

        with torch.no_grad():
            inputs = torch.tensor(tensors, dtype=torch.float32, device=self.device)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
        return float(probs[1]), 'Cavitation' if probs[1] > 0.5 else 'No Cavitation'
