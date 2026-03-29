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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders
from src.models.future_autoencoder import FutureAutoencoder
from src.models.future_latent_predictor import FutureLatentPredictor, FutureLatentPredictorConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_four_slot_predictor import load_autoencoder
from src.utils.noise_schedule import DiffusionNoiseSchedule


@dataclass
class NoisyPredictorTrainConfig:
    ae_checkpoint_path: str = "outputs/checkpoints/ae_four_slot_causal_refine_roberta/ae_best.pt"
    init_from_predictor_checkpoint: str = "outputs/checkpoints/four_slot_predictor_causal_refine_roberta/predictor_best.pt"
    noisy_target_timestep: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 30
    device: str = "auto"
    log_every: int = 100
    checkpoint_dir: str = "outputs/checkpoints/four_slot_noisy_predictor_t15_roberta"
    log_dir: str = "outputs/logs/four_slot_noisy_predictor_t15_roberta"
    save_every_epoch: bool = False
    grad_clip_norm: float | None = 1.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "NoisyPredictorTrainConfig":
        train_config = config.get("training", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in train_config.items() if key in valid_keys}
        return cls(**filtered_config)


def load_predictor(
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PrefixEncoder, FutureLatentPredictor]:
    prefix_encoder = PrefixEncoder(PrefixEncoderConfig.from_dict(config)).to(device)
    predictor = FutureLatentPredictor(FutureLatentPredictorConfig.from_dict(config)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    prefix_encoder.load_state_dict(checkpoint["prefix_encoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    prefix_encoder.freeze_bert_backbone()
    return prefix_encoder, predictor


def run_epoch(
    autoencoder: FutureAutoencoder,
    prefix_encoder: PrefixEncoder,
    predictor: FutureLatentPredictor,
    dataloader: torch.utils.data.DataLoader,
    noise_schedule: DiffusionNoiseSchedule,
    target_timestep: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_every: int,
    grad_clip_norm: float | None,
) -> float:
    is_train = optimizer is not None
    prefix_encoder.train(is_train)
    predictor.train(is_train)

    total_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc="train" if is_train else "val")

    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            target_clean_latent = autoencoder.encode_future(
                future_ids=batch["future_ids"],
                future_mask=batch["future_mask"],
            )
            alpha_bar = noise_schedule.alpha_bars[target_timestep].view(1, 1, 1)
            target_noisy_latent = torch.sqrt(alpha_bar) * target_clean_latent

        if is_train:
            optimizer.zero_grad()

        prefix_states = prefix_encoder(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
        )
        predicted_noisy_latent = predictor(
            prefix_states=prefix_states,
            prefix_mask=batch["prefix_mask"],
        )
        loss = F.mse_loss(predicted_noisy_latent, target_noisy_latent)

        if is_train:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(prefix_encoder.parameters()) + list(predictor.parameters()),
                    grad_clip_norm,
                )
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        if step % log_every == 0 or step == len(dataloader):
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    prefix_encoder: PrefixEncoder,
    predictor: FutureLatentPredictor,
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
            "predictor_state_dict": predictor.state_dict(),
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
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "device": str(device),
                }
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/four_slot_noisy_predictor_t15_roberta.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = DataConfig.from_dict(config)
    predictor_config = FutureLatentPredictorConfig.from_dict(config)
    train_config = NoisyPredictorTrainConfig.from_dict(config)

    device = resolve_device(train_config.device)
    print(f"Using device: {device}")
    _, train_loader, val_loader = build_dataloaders(data_config)

    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=train_config.ae_checkpoint_path,
        device=device,
    )
    prefix_encoder, predictor = load_predictor(
        config=config,
        checkpoint_path=train_config.init_from_predictor_checkpoint,
        device=device,
    )

    # Ensure the predictor matches the config we intend to train.
    predictor.to(device)
    if predictor.config.coarse_slots != predictor_config.coarse_slots:
        raise ValueError("Loaded predictor checkpoint does not match the requested predictor config.")

    noise_schedule = DiffusionNoiseSchedule(
        num_steps=config["model"].get("num_diffusion_steps", train_config.noisy_target_timestep + 1),
        schedule_type=config["model"].get("noise_schedule", "sqrt"),
        timestep_sampling="uniform",
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in list(prefix_encoder.parameters()) + list(predictor.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_path = Path(train_config.log_dir) / "noisy_predictor_train.jsonl"

    for epoch in range(1, train_config.num_epochs + 1):
        print(f"Epoch {epoch}/{train_config.num_epochs}")
        train_loss = run_epoch(
            autoencoder=autoencoder,
            prefix_encoder=prefix_encoder,
            predictor=predictor,
            dataloader=train_loader,
            noise_schedule=noise_schedule,
            target_timestep=train_config.noisy_target_timestep,
            device=device,
            optimizer=optimizer,
            log_every=train_config.log_every,
            grad_clip_norm=train_config.grad_clip_norm,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                autoencoder=autoencoder,
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                dataloader=val_loader,
                noise_schedule=noise_schedule,
                target_timestep=train_config.noisy_target_timestep,
                device=device,
                optimizer=None,
                log_every=train_config.log_every,
                grad_clip_norm=None,
            )

        print(f"train_loss: {train_loss:.4f}")
        print(f"val_loss:   {val_loss:.4f}")
        append_epoch_log(log_path=log_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss, device=device)

        if train_config.save_every_epoch:
            save_checkpoint(
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / f"predictor_epoch_{epoch}.pt",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / "predictor_best.pt",
            )


if __name__ == "__main__":
    main()
