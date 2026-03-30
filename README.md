# Latent Diffusion for Text

This repository keeps one main path:

- train a BART latent autoencoder on `QQP`
- train a latent denoiser on top of that autoencoder
- inspect reconstructions and sampled outputs

The current setup uses frozen `facebook/bart-base` encoder-decoder weights, a Perceiver-style latent bottleneck, and direct latent diffusion.

## Install

```bash
pip install -r requirements.txt
```

## Check the dataset

```bash
python scripts/test_dataset.py
```

## Train the autoencoder

```bash
python -m src.training.train_ae --config configs/ae_bart_latent_qqp.yaml
```

Metrics are appended to `outputs/logs/ae_bart_latent_qqp/ae_train.jsonl`.

## Train the denoiser

Train the autoencoder first so `outputs/checkpoints/ae_bart_latent_qqp/ae_best.pt` exists.

```bash
python -m src.training.train_denoiser --config configs/denoiser_bart_latent_qqp.yaml
```

Metrics are appended to `outputs/logs/denoiser_bart_latent_qqp/denoiser_train.jsonl`.
The denoiser is trained in latent space with `pred_v`, self-conditioning, a `sqrt` noise schedule, and loss-aware timestep sampling.

## Run inference

```bash
python scripts/run_inference.py --config configs/denoiser_bart_latent_qqp.yaml --sample-index 0 --compare-num-steps 10,25,50
```

This prints the `QQP` source, target, AE reconstruction, oracle denoise, and direct denoiser outputs at different reverse-step counts.
