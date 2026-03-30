#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/denoiser_bart_latent_qqp.yaml}"

python -m src.training.train_denoiser --config "${CONFIG_PATH}"
