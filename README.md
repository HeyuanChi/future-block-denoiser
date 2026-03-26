# Future Block Denoiser

Prototype code for an NLP seminar project on non-traditional next-token generation with
future-block latent modeling and refinement.

The current codebase now implements:

- WikiText-2 raw dataset loading and fixed-window slicing
- A future-block autoencoder with a frozen BERT-based encoder
- A lightweight Transformer decoder for token reconstruction
- A plain PyTorch training script for the autoencoder
- A prefix-conditioned latent denoiser trained to predict diffusion noise in latent space
- An optional coarse future latent initializer for few-step latent refinement

## Project Structure

```text
future-block-denoiser/
├── README.md
├── requirements.txt
├── configs/
│   ├── ae.yaml
│   ├── denoiser.yaml
│   └── denoiser_coarse.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── future_latent_initializer.py
│   │   └── future_autoencoder.py
│   ├── training/
│   │   ├── train_ae.py
│   │   ├── train_denoiser.py
│   │   └── train_initializer.py
│   └── utils/
│       ├── metrics.py
│       └── noise_schedule.py
├── scripts/
│   ├── run_inference.py
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

Metrics are appended to `outputs/logs/ae_train.jsonl`.

## Train The Denoiser

Train the autoencoder first so `outputs/checkpoints/ae_best.pt` exists.

```bash
python -m src.training.train_denoiser --config configs/denoiser.yaml
```

Metrics are appended to `outputs/logs/denoiser_train.jsonl`.
The denoiser is trained to predict diffusion noise in latent space.

To continue denoiser training from an existing checkpoint, set
`training.resume_from_checkpoint` in `configs/denoiser.yaml`.

## Coarse-To-Fine Refinement

The repository also supports a coarse-to-fine variant of Stage 2:

1. Train a prefix-conditioned coarse future latent initializer.
2. Train the denoiser to refine around that coarse latent anchor.
3. Run inference with the initializer and the refinement denoiser together.

The sample configuration for this path is `configs/denoiser_coarse.yaml`.

### Train The Coarse Initializer

```bash
python -m src.training.train_initializer --config configs/denoiser_coarse.yaml
```

This writes metrics to `outputs/logs/initializer_train.jsonl` and saves the best
checkpoint to `outputs/checkpoints/initializer_best.pt`.

### Train The Anchor-Based Denoiser

After the initializer has been trained:

```bash
python -m src.training.train_denoiser --config configs/denoiser_coarse.yaml
```

In this mode the denoiser still predicts diffusion noise, but it operates on the
residual around the initializer output instead of starting from pure Gaussian noise.

## Run Inference

```bash
python scripts/run_inference.py --config configs/denoiser.yaml
```

This prints a validation prefix, the ground-truth future block, the AE
reconstruction, and the denoiser-based future prediction.

For the coarse-to-fine variant:

```bash
python scripts/run_inference.py --config configs/denoiser_coarse.yaml
```

This additionally prints the coarse initializer prediction and its latent MSE to the
AE target.
