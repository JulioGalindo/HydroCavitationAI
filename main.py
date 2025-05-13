
"""Command‑line interface for cavitation detector."""
import argparse, sys
import numpy as np
import soundfile as sf
from preprocess import Preprocessor
from vmd import VMD
from features import FeatureExtractor
from train import Trainer
from predict import Predictor
import json, os

def load_signal(path):
    if path.lower().endswith('.wav'):
        signal, fs = sf.read(path)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        return signal, fs
    elif path.lower().endswith('.csv'):
        data = np.loadtxt(path, delimiter=',')
        return data[:,1], 1.0/(data[1,0]-data[0,0])
    else:
        raise ValueError('Unsupported file format.')

def main():
    parser = argparse.ArgumentParser(description='Cavitation Detection System')
    parser.add_argument('--mode', choices=['train','predict'], required=True)
    parser.add_argument('--input', help='Input file or folder')
    parser.add_argument('--model', default='weights_final.pt')
    parser.add_argument('--report', action='store_true', help='Save reports')
    args = parser.parse_args()

    if args.mode == 'predict':
        signal, fs = load_signal(args.input)
        predictor = Predictor(fs, {'alpha':2000,'K':6,'tau':0}, args.model)
        prob, diagnosis = predictor.predict(signal)
        print(f'Cavitation probability: {prob:.2f}')
        print(f'Diagnosis: {diagnosis}')
    elif args.mode == 'train':
        # Expect folder with npz file tensors.npz containing X and y
        data = np.load(args.input)
        X, y = data['X'], data['y']
        in_channels = 1  # placeholder
        trainer = Trainer(in_channels)
        trainer.fit(X, y, args.model)
        if args.report:
            print('Training finished. Weights saved to', args.model)

if __name__ == '__main__':
    main()
