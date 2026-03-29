#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/four_slot_noisy_predictor_t15_roberta.yaml}"

python -m src.training.train_noisy_latent_predictor --config "${CONFIG_PATH}"
