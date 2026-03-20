from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders
from src.models.future_autoencoder import FutureAutoencoder, FutureAutoencoderConfig
from src.utils.metrics import masked_token_cross_entropy


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 5
    device: str = "auto"
    log_every: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    save_every_epoch: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TrainConfig":
        train_config = config.get("training", config)
        return cls(**train_config)


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    return torch.device(device_name)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: FutureAutoencoder,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_every: int,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(dataloader, desc="train" if is_train else "val")
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        _, logits = model(
            future_ids=batch["future_ids"],
            future_mask=batch["future_mask"],
        )
        loss = masked_token_cross_entropy(
            logits=logits,
            target_ids=batch["future_ids"],
            mask=batch["future_mask"],
        )

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if step % log_every == 0 or step == len(dataloader):
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: FutureAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )


def append_epoch_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    device: torch.device,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "device": str(device),
    }
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ae.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = DataConfig.from_dict(config)
    model_config = FutureAutoencoderConfig.from_dict(config)
    train_config = TrainConfig.from_dict(config)

    device = resolve_device(train_config.device)
    print(f"Using device: {device}")
    _, train_loader, val_loader = build_dataloaders(data_config)

    model = FutureAutoencoder(model_config)
    model.freeze_bert_backbone()
    model.to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_path = Path(train_config.log_dir) / "ae_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.num_epochs + 1):
        print(f"Epoch {epoch}/{train_config.num_epochs}")
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            log_every=train_config.log_every,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                log_every=train_config.log_every,
            )

        print(f"train_loss: {train_loss:.4f}")
        print(f"val_loss:   {val_loss:.4f}")
        append_epoch_log(
            log_path=log_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            device=device,
        )

        if train_config.save_every_epoch:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / f"ae_epoch_{epoch}.pt",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / "ae_best.pt",
            )


if __name__ == "__main__":
    main()
