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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders
from src.models.future_autoencoder import FutureAutoencoder, FutureAutoencoderConfig
from src.models.latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from src.models.context_encoder import ContextEncoder, ContextEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.utils.noise_schedule import DiffusionNoiseSchedule


@dataclass
class DenoiserTrainConfig:
    ae_checkpoint_path: str = "outputs/checkpoints/ae_best.pt"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 5
    device: str = "auto"
    log_every: int = 100
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    save_every_epoch: bool = True
    grad_clip_norm: float | None = 1.0
    warmup_ratio: float = 0.05

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DenoiserTrainConfig":
        train_config = config.get("training", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in train_config.items() if key in valid_keys}
        return cls(**filtered_config)


def load_autoencoder(
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> FutureAutoencoder:
    ae_config = FutureAutoencoderConfig.from_dict(config)
    autoencoder = FutureAutoencoder(ae_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.freeze_bert_backbone()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    autoencoder.eval()
    autoencoder.to(device)
    return autoencoder


def run_epoch(
    autoencoder: FutureAutoencoder,
    context_encoder: ContextEncoder,
    denoiser: LatentDenoiser,
    dataloader: torch.utils.data.DataLoader,
    noise_schedule: DiffusionNoiseSchedule,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    log_every: int,
    grad_clip_norm: float | None,
) -> float:
    is_train = optimizer is not None
    context_encoder.train(is_train)
    denoiser.train(is_train)

    total_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc="train" if is_train else "val")

    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            clean_latent = autoencoder.encode_future(
                future_ids=batch["future_ids"],
                future_mask=batch["future_mask"],
            )
            latent_mask = torch.ones(
                clean_latent.size(0),
                clean_latent.size(1),
                device=device,
                dtype=batch["future_mask"].dtype,
            )

        timesteps = noise_schedule.sample_timesteps(batch["future_ids"].size(0))
        noisy_latent, _ = noise_schedule.add_noise(clean_latent, timesteps)

        if is_train:
            optimizer.zero_grad()

        context_ids, context_mask = build_context_inputs(batch)
        context_states = context_encoder(
            context_ids=context_ids,
            context_mask=context_mask,
        )
        self_condition_latent = None
        if denoiser.config.self_conditioning and torch.rand(()) < 0.5:
            with torch.no_grad():
                self_condition_latent = denoiser(
                    noisy_latent=noisy_latent,
                    context_states=context_states,
                    context_mask=context_mask,
                    timesteps=timesteps,
                    future_mask=latent_mask,
                ).detach()

        predicted_clean = denoiser(
            noisy_latent=noisy_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=timesteps,
            future_mask=latent_mask,
            self_condition_latent=self_condition_latent,
        )

        loss_mask = latent_mask.unsqueeze(-1).float()
        squared_error = ((predicted_clean - clean_latent) * loss_mask) ** 2
        per_example_loss = squared_error.sum(dim=(1, 2)) / (loss_mask.sum(dim=(1, 2)) * predicted_clean.size(-1)).clamp_min(1.0)
        loss = per_example_loss.mean()
        if is_train:
            noise_schedule.update_with_losses(timesteps=timesteps, losses=per_example_loss)

        if is_train:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(context_encoder.parameters()) + list(denoiser.parameters()),
                    grad_clip_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_batches += 1

        if step % log_every == 0 or step == len(dataloader):
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def build_context_inputs(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if "suffix_ids" not in batch:
        return batch["prefix_ids"], batch["prefix_mask"]
    context_ids = torch.cat([batch["prefix_ids"], batch["suffix_ids"]], dim=1)
    context_mask = torch.cat([batch["prefix_mask"], batch["suffix_mask"]], dim=1)
    return context_ids, context_mask


def save_checkpoint(
    context_encoder: ContextEncoder,
    denoiser: LatentDenoiser,
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
            "context_encoder_state_dict": context_encoder.state_dict(),
            "denoiser_state_dict": denoiser.state_dict(),
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
    parser.add_argument("--config", type=str, default="configs/denoiser.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = DataConfig.from_dict(config)
    context_config = ContextEncoderConfig.from_dict(config)
    denoiser_config = LatentDenoiserConfig.from_dict(config)
    train_config = DenoiserTrainConfig.from_dict(config)

    device = resolve_device(train_config.device)
    print(f"Using device: {device}")
    _, train_loader, val_loader = build_dataloaders(data_config)

    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=train_config.ae_checkpoint_path,
        device=device,
    )
    context_encoder = ContextEncoder(context_config)
    context_encoder.freeze_bert_backbone()
    context_encoder.to(device)

    denoiser = LatentDenoiser(denoiser_config).to(device)
    noise_schedule = DiffusionNoiseSchedule(
        num_steps=denoiser_config.num_diffusion_steps,
        schedule_type=config["model"].get("noise_schedule", "sqrt"),
        timestep_sampling=config["model"].get("timestep_sampling", "loss_aware"),
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in list(context_encoder.parameters()) + list(denoiser.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_train_steps = max(len(train_loader) * train_config.num_epochs, 1)
    warmup_steps = int(total_train_steps * train_config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_loss = float("inf")
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_path = Path(train_config.log_dir) / "denoiser_train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if warmup_steps > 0:
        print(f"Using linear LR warmup for {warmup_steps} steps ({train_config.warmup_ratio:.1%} of training).")

    for epoch in range(1, train_config.num_epochs + 1):
        print(f"Epoch {epoch}/{train_config.num_epochs}")
        train_loss = run_epoch(
            autoencoder=autoencoder,
            context_encoder=context_encoder,
            denoiser=denoiser,
            dataloader=train_loader,
            noise_schedule=noise_schedule,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            log_every=train_config.log_every,
            grad_clip_norm=train_config.grad_clip_norm,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                autoencoder=autoencoder,
                context_encoder=context_encoder,
                denoiser=denoiser,
                dataloader=val_loader,
                noise_schedule=noise_schedule,
                device=device,
                optimizer=None,
                scheduler=None,
                log_every=train_config.log_every,
                grad_clip_norm=None,
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
                context_encoder=context_encoder,
                denoiser=denoiser,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / f"denoiser_epoch_{epoch}.pt",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                context_encoder=context_encoder,
                denoiser=denoiser,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / "denoiser_best.pt",
            )


if __name__ == "__main__":
    main()
