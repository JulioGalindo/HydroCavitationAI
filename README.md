
# HydroCavitationAI

**HydroCavitationAI** is a Python-based toolkit for real-time cavitation diagnosis in hydraulic turbines (Francis, Pelton, Kaplan).  
It implements the state-of-the-art **SSA-VMD-MSCNN** workflow published by **Li et al., 2024**:

* **SSA (Sparrow Search Algorithm)** automatically tunes Variational Mode Decomposition hyper-parameters.  
* **VMD** extracts cavitation-sensitive intrinsic modes from high-frequency acoustic or vibration signals.  
* **MSCNN** ingests multichannel STFT spectrograms and classifies *Cavitation* vs *No Cavitation* with ≥ 98 % reported accuracy.

## Key Features

- **End-to-end pipeline**: data loading → preprocessing → SSA-VMD decomposition → feature extraction → neural-network training / inference.  
- **Modular, class-only codebase** compatible with Linux and macOS (no Windows-specific paths).  
- **Command-line interface** for training, prediction and evaluation; automatically generates confusion matrix, ROC/AUC plots, and JSON metrics.  
- **Fully reproducible**: deterministic seeds, `config.yaml`, and explicit `requirements.txt`.

## Typical Workflow

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train the model
python main.py --mode train   \
               --input data/tensors.npz \
               --model weights_final.pt \
               --report

# 3. Run inference on a new signal
python main.py --mode predict \
               --input data/sample.wav \
               --model weights_final.pt

## Citation

If you use **HydroCavitationAI** in academic work, please cite:

> Li, X., Wang, Y., & Zhang, H. (2024). *Sparrow search algorithm-optimized variational mode decomposition and multiscale convolutional neural network for cavitation diagnosis in hydraulic turbines*. **Ocean Engineering**, 312, 119055. https://doi.org/10.1016/j.oceaneng.2024.119055

# Cavitation Detector (SSA‑VMD‑MSCNN)

Implementation of Li et al. (2024) approach for real‑time cavitation
diagnosis in hydraulic turbines.

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Training
```bash
python main.py --mode train --input data/tensors.npz --model weights_final.pt --report
```

### Inference
```bash
python main.py --mode predict --input data/sample.wav --model weights_final.pt
```

## Project Structure
See accompanying documentation for module descriptions.

## License
MIT
