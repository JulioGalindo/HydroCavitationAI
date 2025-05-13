# simulate_data.py
"""
SyntheticDataGenerator
----------------------
• Generates synthetic cavitation / non-cavitation signals *and/or*
  builds the STFT tensor expected by `train.py`.
• Offers two CLI sub-commands:
      ▸ synthetic  – create synthetic signals (+ optional tensor)
      ▸ tensor     – convert existing signals to tensors
All logic is encapsulated in a single class; no global functions.
"""

from pathlib import Path
from typing import Tuple, Iterable, List
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import chirp
from preprocess import Preprocessor
from features import FeatureExtractor


class SyntheticDataGenerator:
    """Utility class for creating signals and tensors for HydroCavitationAI."""

    def __init__(
        self,
        fs: float = 5.0e5,
        duration: float = 10.0,
        seg_length: float = 0.1,
        n_fft: int = 256,
        hop: int = 128,
        seed: int | None = 42,
    ):
        self.fs = fs
        self.duration = duration
        self.n_samples = int(fs * duration)
        self.pre = Preprocessor(fs, seg_length=seg_length)
        self.feat = FeatureExtractor(fs, n_fft=n_fft, hop=hop)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ API

    def generate_synthetic_dataset(
        self,
        n_signals: int,
        out_dir: str | Path = "data/synthetic",
        make_tensor: bool = True,
    ) -> None:
        """
        Produce synthetic signals and (optionally) tensors.npz.

        Parameters
        ----------
        n_signals : int
            Number of signals per class.
        out_dir : str | Path
            Output directory; created if missing.
        make_tensor : bool
            If True, also saves tensors.npz (X, y).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        nocav = self._build_non_cavitation(n_signals)
        cav = self._build_cavitation(n_signals)

        np.save(out_dir / "signals_nocav.npy", nocav)
        np.save(out_dir / "signals_cav.npy", cav)
        print(f"Saved raw signals to {out_dir}")

        if make_tensor:
            self.build_tensor_from_arrays(
                nocav,
                cav,
                out_dir / "tensors.npz",
            )

    def build_tensor_from_arrays(
        self,
        signals_nocav: np.ndarray,
        signals_cav: np.ndarray,
        out_file: str | Path = "data/tensors.npz",
    ) -> None:
        """
        Convert two numpy arrays of signals into tensors.npz.

        Each array shape: (N_signals, samples)

        Parameters
        ----------
        signals_nocav : np.ndarray
            Non-cavitation signals.
        signals_cav : np.ndarray
            Cavitation signals.
        out_file : str | Path
            Output .npz path (contains X, y).
        """
        X, y = self._signals_to_tensors(signals_nocav, signals_cav)
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_file, X=X, y=y)
        print(f"Saved {X.shape[0]} tensors to {out_file}")

    def build_tensor_from_directory(
        self,
        input_dir: str | Path,
        out_file: str | Path = "data/tensors.npz",
        pattern_nocav: str = "nocav",
        pattern_cav: str = "cav",
    ) -> None:
        """
        Scan a directory for signal files and build tensors.

        The method classifies files by filename pattern:
            * pattern_nocav  → label 0
            * pattern_cav    → label 1

        Supported extensions: .wav, .npy, .csv

        Parameters
        ----------
        input_dir : str | Path
            Directory containing the signals.
        out_file : str | Path
            Output .npz path.
        pattern_nocav : str
            Substring identifying non-cavitation files.
        pattern_cav : str
            Substring identifying cavitation files.
        """
        input_dir = Path(input_dir)
        files = list(input_dir.glob("*"))
        nocav_files = [f for f in files if pattern_nocav in f.stem]
        cav_files = [f for f in files if pattern_cav in f.stem]

        def load_signal(f: Path) -> np.ndarray:
            if f.suffix.lower() == ".wav":
                sig, _ = sf.read(f)
                return sig.mean(axis=1) if sig.ndim > 1 else sig
            if f.suffix.lower() == ".npy":
                return np.load(f)
            if f.suffix.lower() == ".csv":
                data = np.loadtxt(f, delimiter=",")
                return data[:, 1]
            raise ValueError(f"Unsupported format: {f}")

        signals_nocav = np.stack([load_signal(f) for f in nocav_files])
        signals_cav = np.stack([load_signal(f) for f in cav_files])

        self.build_tensor_from_arrays(signals_nocav, signals_cav, out_file)

    # ------------------------------------------------------- internal logic

    def _build_non_cavitation(self, n: int) -> np.ndarray:
        sigs: List[np.ndarray] = []
        for _ in range(n):
            t = np.arange(self.n_samples) / self.fs
            f0 = self.rng.uniform(5e3, 10e3)
            s = (
                0.8 * np.sin(2 * np.pi * f0 * t)
                + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
            )
            s += self.rng.normal(0, 0.05, self.n_samples)
            sigs.append(s.astype(np.float32))
        return np.stack(sigs)

    def _build_cavitation(self, n: int) -> np.ndarray:
        sigs: List[np.ndarray] = []
        for _ in range(n):
            s = self._build_non_cavitation(1)[0]
            n_bursts = self.rng.integers(50, 200)
            for _ in range(n_bursts):
                start = self.rng.integers(0, self.n_samples - 1500)
                burst = chirp(
                    np.arange(1500) / self.fs,
                    f0=self.rng.uniform(2e5, 5e5),
                    f1=self.rng.uniform(2e5, 5e5),
                    t1=1500 / self.fs,
                    method="linear",
                )
                s[start : start + 1500] += 0.3 * burst
            sigs.append(s.astype(np.float32))
        return np.stack(sigs)

    def _signals_to_tensors(
        self, nocav: np.ndarray, cav: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for cls, arr in enumerate((nocav, cav)):
            for sig in arr:
                segs = self.pre.segment(
                    self.pre.normalize(self.pre.bandpass(sig))
                )
                tensor = self.feat.stft_tensor(segs)[:, None, ...]  # (N,1,H,W)
                X.append(tensor)
                y += [cls] * len(tensor)
        return np.concatenate(X, axis=0), np.asarray(y, dtype=np.int64)


# ------------------------------------------------------------------ CLI

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic signals or build tensors for HydroCavitationAI."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # synthetic
    p_gen = sub.add_parser("synthetic", help="create synthetic dataset")
    p_gen.add_argument("--n", type=int, default=500, help="signals per class")
    p_gen.add_argument("--out", default="data/synthetic", help="output directory")
    p_gen.add_argument(
        "--no-tensor", action="store_true", help="skip tensor generation"
    )

    # tensor from existing signals
    p_ten = sub.add_parser("tensor", help="build tensor from existing signals")
    p_ten.add_argument(
        "--in", dest="indir", required=True, help="directory with signal files"
    )
    p_ten.add_argument(
        "--out", default="data/tensors.npz", help="output tensors .npz"
    )
    p_ten.add_argument("--pat-nocav", default="nocav", help="pattern non-cavitation")
    p_ten.add_argument("--pat-cav", default="cav", help="pattern cavitation")

    args = parser.parse_args()
    gen = SyntheticDataGenerator()

    if args.cmd == "synthetic":
        gen.generate_synthetic_dataset(
            n_signals=args.n, out_dir=args.out, make_tensor=not args.no_tensor
        )
    elif args.cmd == "tensor":
        gen.build_tensor_from_directory(
            input_dir=args.indir,
            out_file=args.out,
            pattern_nocav=args.pat_nocav,
            pattern_cav=args.pat_cav,
        )


if __name__ == "__main__":
    _cli()
