#!/usr/bin/env python3
"""
simulate_data.py
================
Create synthetic cavitation / non-cavitation signals for HydroCavitationAI or
convert existing recordings to STFT tensors.

Highlights
----------
• GPU-aware (Apple-MPS) with CPU fallback.
• Chunked generation → constant RAM; streamed mem-map write.
• tqdm progress bars.
• Different durations per class (--dur-nocav / --dur-cav).
• Train/test split (tensors_train.npz, tensors_test.npz).
"""

from __future__ import annotations

import sys
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from torch import pi
from numpy.lib.format import open_memmap

# tqdm is optional – fall back to plain range if missing
try:
    from tqdm import trange
except ModuleNotFoundError:        # noqa: D401
    def trange(*a, **k):           # type: ignore
        return range(*a, **k)

from preprocess import Preprocessor
from features import FeatureExtractor

# ------------------------- device ----------------------------------- #
_HAS_MPS = torch.backends.mps.is_available() if sys.platform == "darwin" else False
DEVICE_DEFAULT = torch.device("mps" if _HAS_MPS else "cpu")
DTYPE = torch.float32

# ------------------------- tiny timers ------------------------------- #
def tic(msg: str):
    print(f"[{msg:<25}] start …", flush=True)
    return msg, perf_counter()


def toc(t0):
    msg, t0 = t0
    print(f"[{msg:<25}] done in {perf_counter() - t0:6.2f} s", flush=True)


# ===================================================================== #
#  SyntheticDataGenerator
# ===================================================================== #
class SyntheticDataGenerator:
    """
    Generate synthetic datasets; safe on memory and GPU-aware.
    """

    # --------------------- ctor -------------------------------------- #
    def __init__(
        self,
        *,
        fs: float,
        duration_nocav: float,
        duration_cav: float,
        band_lo: float,
        band_hi: float | None,
        seg_length: float,
        n_fft: int,
        hop: int,
        seed: int,
        device: torch.device,
    ):
        self.fs = fs
        self.duration_nocav = duration_nocav
        self.duration_cav = duration_cav
        self.n_samples_nocav = int(fs * duration_nocav)
        self.n_samples_cav = int(fs * duration_cav)

        nyquist_margin = 0.45 * fs
        band_hi = 4.5e5 if band_hi is None else band_hi
        self.band = (band_lo, min(band_hi, nyquist_margin))

        self.pre = Preprocessor(fs, band=self.band, seg_length=seg_length)
        self.feat = FeatureExtractor(fs, n_fft=n_fft, hop=hop)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(seed)
        self.device = device

    # --------------------- helpers ----------------------------------- #
    def _base(self, batch: int, n_samples: int) -> torch.Tensor:
        """Harmonic + Gaussian noise."""
        t = torch.linspace(0.0, 1.0, n_samples, dtype=DTYPE,
                           device=self.device).expand(batch, -1)
        f0 = torch.rand(batch, device=self.device) * 5e3 + 5e3
        s = 0.8 * torch.sin(2 * pi * f0.unsqueeze(1) * t)
        s += 0.1 * torch.sin(4 * pi * f0.unsqueeze(1) * t)
        s += 0.1 * torch.sin(6 * pi * f0.unsqueeze(1) * t)
        s += 0.05 * torch.randn_like(s)
        return s

    # --------------------- public  synthetic ------------------------- #
    def generate(
        self,
        *,
        n_signals: int,
        out_dir: Path,
        chunk: int,
        dtype: str,
        legacy: bool,
        test_ratio: float,
    ):
        """
        Build raw signals and split tensors into train / test.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # mem-maps avoid >RAM copy
        p_nc = out_dir / "signals_nocav.npy"
        p_c  = out_dir / "signals_cav.npy"
        shape_nc = (n_signals, self.n_samples_nocav)
        shape_c  = (n_signals, self.n_samples_cav)

        if legacy:
            buf_nc = np.empty(shape_nc, dtype=dtype)
            buf_c  = np.empty(shape_c,  dtype=dtype)
        else:
            buf_nc = open_memmap(p_nc, "w+", dtype, shape_nc)
            buf_c  = open_memmap(p_c,  "w+", dtype, shape_c)

        # ---- non-cav
        tic_nc = tic("non-cav generation")
        idx = 0
        for _ in trange((n_signals + chunk - 1) // chunk, desc="non-cav"):
            bs = min(chunk, n_signals - idx)
            buf_nc[idx:idx + bs] = self._base(bs, self.n_samples_nocav) \
                .cpu().numpy().astype(dtype)
            idx += bs
        if not legacy: buf_nc.flush()
        toc(tic_nc)

        # ---- cav
        tic_c = tic("cav generation")
        idx = 0
        for _ in trange((n_signals + chunk - 1) // chunk, desc="cav"):
            bs = min(chunk, n_signals - idx)
            batch = self._base(bs, self.n_samples_cav)
            n_bursts = torch.randint(50, 200, (bs,), device=self.device)
            for i in range(bs):
                for _ in range(int(n_bursts[i])):
                    start = torch.randint(0, self.n_samples_cav - 1500,
                                          (1,), device=self.device)
                    idx_range = torch.arange(1500, device=self.device) + start
                    freq = torch.rand(1, device=self.device) * 3e5 + 2e5
                    burst = 0.3 * torch.sin(2 * pi * freq * idx_range / self.fs)
                    batch[i, idx_range] += burst
            buf_c[idx:idx + bs] = batch.cpu().numpy().astype(dtype)
            idx += bs
        if not legacy: buf_c.flush()
        toc(tic_c)

        if legacy:
            np.save(p_nc, buf_nc)
            np.save(p_c,  buf_c)

        print("[OK] raw signals stored in", out_dir)
        self._build_tensors(buf_nc, buf_c, out_dir, test_ratio, dtype, legacy)

    # ---------------- build tensors & split -------------------------- #
    def _build_tensors(
        self,
        nc: np.ndarray,
        cav: np.ndarray,
        out_dir: Path,
        test_ratio: float,
        dtype: str,
        legacy: bool
    ):
        tic_ts = tic("tensor build")
        X, y = self._signals_to_tensors(nc, cav)
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )
        np.savez(out_dir / "tensors_train.npz", X=X_tr, y=y_tr)
        np.savez(out_dir / "tensors_test.npz",  X=X_te, y=y_te)
        toc(tic_ts)
        print("[OK] train / test tensors saved to", out_dir)

    # ---------------- STFT pipeline (with bars) ---------------------- #
    def _signals_to_tensors(
        self, nc: np.ndarray, cav: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        tensors, labels = [], []
        for cls, arr, name in ((0, nc, "non-cav"), (1, cav, "cav")):
            outer = trange(len(arr), desc=f"{name}  signals", leave=True)
            for sig_idx in outer:
                segs = self.pre.segment(
                    self.pre.normalize(self.pre.bandpass(arr[sig_idx]))
                )
                inner = trange(segs.shape[0], desc="  segments",
                               leave=False, position=1, ncols=70)
                spec_list = [
                    self.feat.stft_tensor(segs[i:i + 1])[0]
                    for i in inner
                ]
                tensor = np.stack(spec_list)[:, None, ...]  # (N,1,H,W)
                tensors.append(tensor)
                labels.extend([cls] * len(tensor))
        return np.concatenate(tensors, axis=0), np.asarray(labels, np.int64)


# ===================================================================== #
#  CLI
# ===================================================================== #
def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Synthetic cavitation generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    # synthetic -------------------------------------------------------- #
    s = sub.add_parser("synthetic")
    s.add_argument("--n", type=int, default=100, help="signals per class")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--fs", type=float, default=1.0e6)
    s.add_argument("--duration", type=float, default=10.0,
                   help="default duration for non-cav (s)")
    s.add_argument("--dur-cav", type=float, default=None,
                   help="duration cav signals (s)")
    s.add_argument("--dur-nocav", type=float, default=None,
                   help="duration non-cav signals (s)")
    s.add_argument("--chunk", type=int, default=100)
    s.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    s.add_argument("--band-lo", type=float, default=1.0e5)
    s.add_argument("--band-hi", type=float, default=None)
    s.add_argument("--legacy-save", action="store_true")
    s.add_argument("--no-tensor", action="store_true")
    s.add_argument("--test-ratio", type=float, default=0.2)
    dev = s.add_mutually_exclusive_group()
    dev.add_argument("--gpu", action="store_true")
    dev.add_argument("--cpu", action="store_true")

    # tensor ----------------------------------------------------------- #
    t = sub.add_parser("tensor")
    t.add_argument("--in", dest="indir", required=True)
    t.add_argument("--out", default="data/tensors.npz")
    t.add_argument("--pat-nocav", default="nocav")
    t.add_argument("--pat-cav",  default="cav")

    args = p.parse_args()

    if args.cmd == "synthetic":
        if args.gpu and not _HAS_MPS:
            p.error("GPU requested but MPS unavailable")
        device = torch.device("mps" if (_HAS_MPS and not args.cpu) or args.gpu
                              else "cpu")

        gen = SyntheticDataGenerator(
            fs=args.fs,
            duration_nocav=args.dur_nocav or args.duration,
            duration_cav=args.dur_cav or args.duration,
            band_lo=args.band_lo,
            band_hi=args.band_hi,
            seg_length=0.1,
            n_fft=256,
            hop=128,
            seed=42,
            device=device,
        )
        gen.generate(
            n_signals=args.n,
            out_dir=Path(args.out),
            chunk=args.chunk,
            dtype=args.dtype,
            legacy=args.legacy_save,
            test_ratio=args.test_ratio,
        )

    else:   # convert existing recordings (unchanged)
        gen = SyntheticDataGenerator(
            fs=args.fs,                       # target fs
            duration_nocav=1, duration_cav=1, # dummy
            band_lo=1e5, band_hi=4.5e5,
            seg_length=0.1, n_fft=256, hop=128,
            seed=42, device=DEVICE_DEFAULT,
        )
        gen.build_tensor_from_directory(
            input_dir=Path(args.indir),
            out_file=Path(args.out),
            pattern_nocav=args.pat_nocav,
            pattern_cav=args.pat_cav,
        )

if __name__ == "__main__":
    main()
