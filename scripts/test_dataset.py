from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders


def main() -> None:
    with open("configs/ae.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = DataConfig.from_dict(config)
    tokenizer, train_loader, val_loader = build_dataloaders(data_config)

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"prefix_ids shape: {tuple(batch['prefix_ids'].shape)}")
    print(f"future_ids shape: {tuple(batch['future_ids'].shape)}")
    print(f"prefix_mask shape: {tuple(batch['prefix_mask'].shape)}")
    print(f"future_mask shape: {tuple(batch['future_mask'].shape)}")

    prefix_text = tokenizer.decode(batch["prefix_ids"][0].tolist(), skip_special_tokens=True)
    future_text = tokenizer.decode(batch["future_ids"][0].tolist(), skip_special_tokens=True)
    print(f"prefix example: {prefix_text}")
    print(f"future example: {future_text}")


if __name__ == "__main__":
    main()
