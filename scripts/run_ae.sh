#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/ae_four_slot_causal_refine_roberta.yaml}"

python -m src.training.train_ae --config "${CONFIG_PATH}"
