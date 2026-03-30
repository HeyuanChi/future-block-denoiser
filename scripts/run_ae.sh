#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/ae_bart_latent_qqp.yaml}"

python -m src.training.train_ae --config "${CONFIG_PATH}"
