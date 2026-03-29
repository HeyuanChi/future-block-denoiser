#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/four_slot_denoiser_causal_refine_roberta.yaml}"

python -m src.training.train_denoiser --config "${CONFIG_PATH}"
