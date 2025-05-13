#!/usr/bin/env python3
"""Visualize raw signals (.npy) or STFT tensors (.npz).

Usage examples
--------------
# Plot a single raw signal
python visualize.py --file data/demo2/signals_cav.npy --signal 0

# Plot the first 3 signals in one figure
python visualize.py --file data/demo2/signals_nocav.npy --multi 3

# Plot spectrogram #10 contained in tensors_train.npz
python visualize.py --file data/demo2/tensors_train.npz --spectrogram 10

The module can also be imported and the ``Visualizer`` class used directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Helper class for visualising .npy signals and .npz tensors."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = np.load(self.path, mmap_mode="r")
        # .npz returns a NpzFile object; .npy returns ndarray
        if isinstance(self.data, np.lib.npyio.NpzFile):
            self.is_npz = True
            self.X = self.data["X"]
            self.y = self.data["y"]
        else:
            self.is_npz = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_signal(self, idx: int = 0, save: Optional[str] = None) -> None:
        """Plot a single raw signal from .npy.

        Parameters
        ----------
        idx  : Index of the signal to display.
        save : Optional path to save the figure instead of showing.
        """
        if self.is_npz:
            raise RuntimeError("This file is an NPZ of tensors. Use plot_spectrogram().")
        if idx >= self.data.shape[0]:
            raise IndexError("Signal index out of range.")

        plt.figure(figsize=(10, 3))
        plt.plot(self.data[idx])
        plt.title(f"Signal #{idx}")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        if save:
            plt.savefig(save, dpi=200)
        else:
            plt.show()

    def plot_multiple(self, n: int = 3, save: Optional[str] = None) -> None:
        """Plot `n` signals in a single figure."""
        if self.is_npz:
            raise RuntimeError("This file is an NPZ of tensors. Use plot_spectrogram().")
        n = min(n, self.data.shape[0])
        plt.figure(figsize=(10, 4))
        for i in range(n):
            plt.plot(self.data[i], label=f"sig {i}")
        plt.title(f"First {n} signals in {self.path.name}")
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(save, dpi=200)
        else:
            plt.show()

    def plot_spectrogram(self, idx: int = 0, save: Optional[str] = None) -> None:
        """Plot a spectrogram from tensors in .npz.

        Parameters
        ----------
        idx  : Sample index within X.
        save : Optional output filename.
        """
        if not self.is_npz:
            raise RuntimeError("This file is a NPY with raw signals. Use plot_signal().")
        if idx >= self.X.shape[0]:
            raise IndexError("Spectrogram index out of range.")

        spec = self.X[idx, 0]  # (H, W)
        plt.figure(figsize=(6, 4))
        plt.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        plt.title(f"Spectrogram #{idx}  (label={int(self.y[idx])})")
        plt.colorbar(label="Magnitude")
        plt.xlabel("Time bins")
        plt.ylabel("Freq bins")
        if save:
            plt.savefig(save, dpi=200)
        else:
            plt.show()


# ---------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize .npy or .npz data.")
    parser.add_argument("--file", required=True, help="Path to .npy or .npz file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--signal", type=int, help="Plot single signal by index")
    group.add_argument("--multi", type=int, help="Plot N first signals")
    group.add_argument("--spectrogram", type=int, help="Plot spectrogram index")

    parser.add_argument("--save", help="Save figure to file instead of showing")
    return parser.parse_args()


def main():
    args = _parse_args()
    vis = Visualizer(args.file)

    try:
        if args.signal is not None:
            vis.plot_signal(args.signal, args.save)
        elif args.multi is not None:
            vis.plot_multiple(args.multi, args.save)
        elif args.spectrogram is not None:
            vis.plot_spectrogram(args.spectrogram, args.save)
    finally:
        # Close npz file to release file handle
        if isinstance(vis.data, np.lib.npyio.NpzFile):
            vis.data.close()


if __name__ == "__main__":
    main()
