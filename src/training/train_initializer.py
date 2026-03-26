from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders
from src.models.future_autoencoder import FutureAutoencoder
from src.models.future_latent_initializer import FutureLatentInitializer
from src.models.future_latent_initializer import FutureLatentInitializerConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_denoiser import load_autoencoder


@dataclass
class InitializerTrainConfig:
    ae_checkpoint_path: str = "outputs/checkpoints/ae_best.pt"
    resume_from_checkpoint: str | None = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 20
    device: str = "auto"
    log_every: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    save_every_epoch: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "InitializerTrainConfig":
        train_config = config.get("training", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in train_config.items() if key in valid_keys}
        return cls(**filtered_config)


def run_epoch(
    autoencoder: FutureAutoencoder,
    prefix_encoder: PrefixEncoder,
    initializer: FutureLatentInitializer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_every: int,
) -> float:
    is_train = optimizer is not None
    prefix_encoder.train(is_train)
    initializer.train(is_train)

    total_loss = 0.0
    total_batches = 0

    from tqdm import tqdm

    iterator = tqdm(dataloader, desc="train" if is_train else "val")
    for step, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            clean_latent = autoencoder.encode_future(
                future_ids=batch["future_ids"],
                future_mask=batch["future_mask"],
            )

        if is_train:
            optimizer.zero_grad()

        prefix_states = prefix_encoder(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
        )
        coarse_latent = initializer(
            prefix_states=prefix_states,
            prefix_mask=batch["prefix_mask"],
            future_mask=batch["future_mask"],
        )

        loss_mask = batch["future_mask"].unsqueeze(-1).float()
        loss = F.mse_loss(coarse_latent * loss_mask, clean_latent * loss_mask, reduction="sum")
        normalizer = (loss_mask.sum() * coarse_latent.size(-1)).clamp_min(1.0)
        loss = loss / normalizer

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        if step % log_every == 0 or step == len(dataloader):
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    prefix_encoder: PrefixEncoder,
    initializer: FutureLatentInitializer,
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
            "prefix_encoder_state_dict": prefix_encoder.state_dict(),
            "initializer_state_dict": initializer.state_dict(),
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
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "device": str(device),
    }
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def load_initializer_checkpoint(
    checkpoint_path: str,
    prefix_encoder: PrefixEncoder,
    initializer: FutureLatentInitializer,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, float, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    prefix_encoder.load_state_dict(checkpoint["prefix_encoder_state_dict"])
    initializer.load_state_dict(checkpoint["initializer_state_dict"])

    optimizer_loaded = False
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_loaded = True
        except ValueError:
            optimizer_loaded = False

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("val_loss", float("inf")))
    return start_epoch, best_val_loss, optimizer_loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/denoiser.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = DataConfig.from_dict(config)
    prefix_config = PrefixEncoderConfig.from_dict(config)
    initializer_config = FutureLatentInitializerConfig.from_dict(config)
    train_config = InitializerTrainConfig.from_dict(config)

    device = resolve_device(train_config.device)
    print(f"Using device: {device}")
    _, train_loader, val_loader = build_dataloaders(data_config)

    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=train_config.ae_checkpoint_path,
        device=device,
    )
    prefix_encoder = PrefixEncoder(prefix_config).to(device)
    prefix_encoder.freeze_bert_backbone()

    initializer = FutureLatentInitializer(initializer_config).to(device)

    trainable_parameters = [
        parameter
        for parameter in list(prefix_encoder.parameters()) + list(initializer.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_path = Path(train_config.log_dir) / "initializer_train.jsonl"

    if train_config.resume_from_checkpoint is not None:
        start_epoch, best_val_loss, optimizer_loaded = load_initializer_checkpoint(
            checkpoint_path=train_config.resume_from_checkpoint,
            prefix_encoder=prefix_encoder,
            initializer=initializer,
            optimizer=optimizer,
            device=device,
        )
        print(f"Resuming from checkpoint: {train_config.resume_from_checkpoint}")
        print(f"Starting at epoch {start_epoch}")
        if not optimizer_loaded:
            print("Optimizer state could not be restored exactly. Continuing with a fresh optimizer state.")

    for epoch in range(start_epoch, train_config.num_epochs + 1):
        print(f"Epoch {epoch}/{train_config.num_epochs}")
        train_loss = run_epoch(
            autoencoder=autoencoder,
            prefix_encoder=prefix_encoder,
            initializer=initializer,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            log_every=train_config.log_every,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                autoencoder=autoencoder,
                prefix_encoder=prefix_encoder,
                initializer=initializer,
                dataloader=val_loader,
                device=device,
                optimizer=None,
                log_every=train_config.log_every,
            )

        print(f"train_loss: {train_loss:.4f}")
        print(f"val_loss:   {val_loss:.4f}")
        append_epoch_log(log_path=log_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss, device=device)

        if train_config.save_every_epoch:
            save_checkpoint(
                prefix_encoder=prefix_encoder,
                initializer=initializer,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / f"initializer_epoch_{epoch}.pt",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                prefix_encoder=prefix_encoder,
                initializer=initializer,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / "initializer_best.pt",
            )


if __name__ == "__main__":
    main()
