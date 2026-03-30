# Future Block Denoiser

Current working direction:

- `QQP` seq2seq paraphrase data
- a frozen `facebook/bart-base` language autoencoder with a perceiver latent bottleneck
- direct latent diffusion on top of that AE

The codebase is intentionally trimmed to this main path:

- train a BART latent AE on `QQP`
- train a direct latent denoiser against AE latents
- inspect reconstructions and denoised generations

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
python -m src.training.train_ae --config configs/ae_bart_latent_qqp.yaml
```

Metrics are appended to `outputs/logs/ae_bart_latent_qqp/ae_train.jsonl`.

## Train The Denoiser

Train the autoencoder first so `outputs/checkpoints/ae_bart_latent_qqp/ae_best.pt` exists.

```bash
python -m src.training.train_denoiser --config configs/denoiser_bart_latent_qqp.yaml
```

Metrics are appended to `outputs/logs/denoiser_bart_latent_qqp/denoiser_train.jsonl`.
The denoiser is trained in latent space with `x0` prediction, self-conditioning,
`sqrt` noise schedule, and loss-aware timestep sampling.

## Run Inference

```bash
python scripts/run_inference.py --config configs/denoiser_bart_latent_qqp.yaml --sample-index 0 --compare-num-steps 10,25,50
```

This prints the `QQP` source, target, AE reconstruction, oracle denoise, and
direct denoiser generations at different reverse-step counts.
