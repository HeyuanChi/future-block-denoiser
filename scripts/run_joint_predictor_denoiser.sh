#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/joint_predictor_denoiser_roberta.yaml}"

python -m src.training.train_joint_predictor_denoiser --config "${CONFIG_PATH}"
