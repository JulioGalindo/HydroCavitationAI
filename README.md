# HydroCavitationAI – SSA‑VMD‑MSCNN Toolkit

Real‑time cavitation diagnosis for Francis / Pelton / Kaplan turbines, reproducing Li et al. (2024) 98 % accuracy pipeline.

## Theory

1. **SSA** tunes VMD hyper‑parameters (α, K, τ) to maximise energy concentration of cavitation modes.  
2. **VMD** decomposes the high‑frequency signal into K intrinsic mode functions (IMFs).  
3. The highest‑frequency three IMFs are converted to STFT images → **3‑channel tensor**.  
4. **MSCNN** classifies each 4‑revolution window (*cav* / *no‑cav*).

## CLI (main.py)

```bash
python main.py generate --out data --n 40 --duration 60 --rpm 300 --fs 1e6
python main.py train    --root data          --epochs 40 --batch 64
python main.py predict  --input my.wav       --model weights.pth
python main.py analyze  --signal long.wav    --model weights.pth --csv timeline.csv
```

| Sub‑cmd   | Key options | Description |
|-----------|-------------|-------------|
| generate  | `--n, --duration, --rpm, --fs, --dtype, --cpu` | Build synthetic mixed dataset; GPU on M‑series by default |
| train     | `--epochs, --batch, --lr, --cpu` | Train MSCNN on VMD blocks; mixed‑precision AMP |
| predict   | `--input (file/dir), --model, --cpu` | Prob. output for each file |
| analyze   | `--signal, --labels, --csv` | Probability timeline + accuracy vs ground truth |

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # numpy, scipy, torch>=2.1, tqdm
```

Apple Silicon users gain automatic Metal/MPS acceleration; others fall back to CPU.

## Citation

Li, X., Wang, Y., Zhang, H. (2024). *SSA‑VMD‑MSCNN for cavitation diagnosis*. **Ocean Engineering**, 312, 119055. https://doi.org/10.1016/j.oceaneng.2024.119055