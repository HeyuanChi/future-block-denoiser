# Future Block Denoiser

Minimal Stage 1 prototype for an NLP seminar project on non-traditional next-token generation.

The current codebase only implements:

- WikiText-2 raw dataset loading and fixed-window slicing
- A future-block autoencoder with a frozen BERT-based encoder
- A lightweight Transformer decoder for token reconstruction
- A plain PyTorch training script for the autoencoder

## Project Structure

```text
future-block-denoiser/
├── README.md
├── requirements.txt
├── configs/
│   ├── ae.yaml
│   └── denoiser.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── future_autoencoder.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_ae.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── scripts/
│   └── test_dataset.py
└── outputs/
    ├── checkpoints/
    └── logs/
```

## Install

```bash
pip install -r requirements.txt
```

## Test The Dataset

```bash
python scripts/test_dataset.py
```

## Train The Autoencoder

```bash
python -m src.training.train_ae --config configs/ae.yaml
```
