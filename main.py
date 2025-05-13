
"""Unified CLI for HydroCavitationAI
====================================
Sub‑commands
------------
generate   – synthetic data generator (SSA‑VMD ready)
train      – block‑level MSCNN trainer
predict    – probability for raw file
analyze    – timeline over long recording

Run `python main.py --help` for details.
"""
from __future__ import annotations
import argparse, platform, sys, json, numpy as np, torch
from pathlib import Path

# local imports
from simulate_interval import generate as gen_data
from train_blocks import train as train_blocks
from block_dataset_vmd import BlockDatasetVMD   # just to confirm presence
from predict_vmd import PredictorVMD
import analyze  # analyze.main

_IS_APPLE = platform.system() == "Darwin" and platform.machine().startswith("arm")
_HAS_MPS  = torch.backends.mps.is_available()

# ------------------------------------------------------------------------- #
# helpers
def _dev(use_cpu: bool) -> str:
    return "cpu" if use_cpu else ("mps" if _IS_APPLE and _HAS_MPS else "cpu")

# ------------------------------------------------------------------------- #
def cmd_generate(args):
    dev=None if not args.cpu else "cpu"
    gen_data(Path(args.out), args.n, args.duration, args.rpm,
             args.fs, args.dtype, device=dev)

def cmd_train(args):
    dev=_dev(args.cpu)
    train_blocks(Path(args.root), args.rpm, args.fs,
                 epochs=args.epochs, batch=args.batch,
                 lr=args.lr, patience=args.patience,
                 device=dev)

def cmd_predict(args):
    dev=_dev(args.cpu)
    pred = PredictorVMD(Path(args.model), Path(args.cfg),
                        args.fs, dev)
    import soundfile as sf
    path = Path(args.input)
    if path.is_dir():
        for f in sorted(path.iterdir()):
            if f.suffix in {".wav",".npy"}:
                x,fs = (sf.read(f)[0] if f.suffix==".wav" else (np.load(f), args.fs))
                p = pred.prob(x)
                print(f"{f.name:30}  prob={p:.3f}  {'cav' if p>=0.5 else 'no-cav'}")
    else:
        import soundfile as sf
        x,fs = (sf.read(path)[0] if path.suffix==".wav" else (np.load(path), args.fs))
        p = pred.prob(x)
        print(f"{path.name:30}  prob={p:.3f}  {'cav' if p>=0.5 else 'no-cav'}")

def cmd_analyze(args):
    sys.argv = ["analyze"] + sum(([f"--{k}", str(v)] for k,v in vars(args).items() if v and k not in {"func"}), [])
    analyze.main()   # re‑enter analyze CLI

# ------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Unified HydroCavitationAI CLI")
    sub = ap.add_subparsers(dest="sub")

    # generate
    g = sub.add_parser("generate", help="synthetic data generator")
    g.add_argument("--out", default="data/mixed")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--duration", type=float, default=12)
    g.add_argument("--rpm", type=float, default=300)
    g.add_argument("--fs", type=float, default=1_000_000)
    g.add_argument("--dtype", choices=["float16","float32","float64"], default="float16")
    g.add_argument("--cpu", action="store_true")
    g.set_defaults(func=cmd_generate)

    # train
    t = sub.add_parser("train", help="train MSCNN on VMD blocks")
    t.add_argument("--root", default="data/mixed")
    t.add_argument("--rpm", type=float, default=300)
    t.add_argument("--fs", type=float, default=1_000_000)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--cpu", action="store_true")
    t.set_defaults(func=cmd_train)

    # predict
    p = sub.add_parser("predict", help="probability for one file or folder")
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="weights_blocks.pth")
    p.add_argument("--cfg", default="ssa_config.json")
    p.add_argument("--fs", type=float, default=1_000_000)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_predict)

    # analyze
    a = sub.add_parser("analyze", help="timeline analyzer over long signal")
    a.add_argument("--signal", required=True)
    a.add_argument("--labels")
    a.add_argument("--rpm", type=float, default=300)
    a.add_argument("--fs", type=float, default=1_000_000)
    a.add_argument("--model", default="weights_blocks.pth")
    a.add_argument("--csv")
    a.add_argument("--cpu", action="store_true")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    if not hasattr(args, "func"):
        ap.print_help(); sys.exit(1)
    args.func(args)

if __name__ == "__main__": main()
