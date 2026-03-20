# Future Block Denoiser

Seminar project on multi-token future block prediction with prefix-conditioned iterative latent denoising.

## Goal
This project studies whether a fixed future token block can be generated with fewer sequential steps than standard autoregressive decoding by iteratively denoising a learned latent representation.

## Current plan
- Stage 1: train a BERT-initialized future-block autoencoder
- Stage 2: train a prefix-conditioned latent denoiser
- Compare against autoregressive and direct block prediction baselines

## Structure
- `src/data`: dataset and preprocessing
- `src/models`: model definitions
- `src/training`: training scripts
- `configs`: experiment configs
- `outputs`: checkpoints and logs