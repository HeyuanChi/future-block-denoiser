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
from src.models.context_encoder import ContextEncoder, ContextEncoderConfig
from src.models.future_autoencoder import FutureAutoencoder
from src.models.future_latent_predictor import FutureLatentPredictor, FutureLatentPredictorConfig
from src.models.latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_denoiser import build_context_inputs, load_autoencoder
from src.utils.metrics import masked_token_cross_entropy
from src.utils.noise_schedule import DiffusionNoiseSchedule


@dataclass
class JointTrainConfig:
    ae_checkpoint_path: str = "outputs/checkpoints/ae_four_slot_causal_refine_roberta/ae_best.pt"
    predictor_checkpoint_path: str = "outputs/checkpoints/four_slot_predictor_causal_refine_roberta/predictor_best.pt"
    denoiser_checkpoint_path: str = "outputs/checkpoints/four_slot_denoiser_causal_refine_roberta/denoiser_best.pt"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 20
    device: str = "auto"
    log_every: int = 100
    checkpoint_dir: str = "outputs/checkpoints/predictor_init_denoiser_joint"
    log_dir: str = "outputs/logs/predictor_init_denoiser_joint"
    save_every_epoch: bool = False
    grad_clip_norm: float | None = 1.0
    warmup_ratio: float = 0.05
    predictor_decode_loss_weight: float = 0.0
    target_denoise_loss_weight: float = 1.0
    predictor_refine_loss_weight: float = 1.0
    predictor_refine_max_t: int = 15

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "JointTrainConfig":
        train_config = config.get("training", config)
        valid_keys = {field.name for field in fields(cls)}
        filtered_config = {key: value for key, value in train_config.items() if key in valid_keys}
        return cls(**filtered_config)


def load_predictor_components(
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


def load_denoiser_components(
    config: dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ContextEncoder, LatentDenoiser]:
    context_encoder = ContextEncoder(ContextEncoderConfig.from_dict(config)).to(device)
    denoiser = LatentDenoiser(LatentDenoiserConfig.from_dict(config)).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    context_encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    denoiser.load_state_dict(checkpoint["denoiser_state_dict"])
    context_encoder.freeze_bert_backbone()
    return context_encoder, denoiser


def sample_timesteps_with_cap(
    noise_schedule: DiffusionNoiseSchedule,
    batch_size: int,
    max_timestep: int,
) -> torch.Tensor:
    max_timestep = min(max_timestep, noise_schedule.num_steps - 1)
    if max_timestep <= 0:
        return torch.zeros(batch_size, device=noise_schedule.device, dtype=torch.long)
    return torch.randint(0, max_timestep + 1, (batch_size,), device=noise_schedule.device)


def build_self_condition_latent(
    denoiser: LatentDenoiser,
    noisy_latent: torch.Tensor,
    context_states: torch.Tensor,
    context_mask: torch.Tensor,
    timesteps: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor | None:
    if not denoiser.config.self_conditioning or torch.rand(()) >= 0.5:
        return None
    with torch.no_grad():
        return denoiser(
            noisy_latent=noisy_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=timesteps,
            future_mask=latent_mask,
        ).detach()


def masked_latent_mse(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    loss_mask = latent_mask.unsqueeze(-1).float()
    squared_error = ((predicted_latent - target_latent) * loss_mask) ** 2
    per_example_loss = squared_error.sum(dim=(1, 2)) / (
        loss_mask.sum(dim=(1, 2)) * predicted_latent.size(-1)
    ).clamp_min(1.0)
    return per_example_loss.mean()


def run_epoch(
    autoencoder: FutureAutoencoder,
    prefix_encoder: PrefixEncoder,
    predictor: FutureLatentPredictor,
    context_encoder: ContextEncoder,
    denoiser: LatentDenoiser,
    dataloader: torch.utils.data.DataLoader,
    noise_schedule: DiffusionNoiseSchedule,
    train_config: JointTrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    prefix_encoder.train(is_train)
    predictor.train(is_train)
    context_encoder.train(is_train)
    denoiser.train(is_train)

    total_loss = 0.0
    total_predictor_loss = 0.0
    total_target_denoise_loss = 0.0
    total_predictor_refine_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(dataloader, desc="train" if is_train else "val")
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)

        with torch.no_grad():
            target_latent = autoencoder.encode_future(
                future_ids=batch["future_ids"],
                future_mask=batch["future_mask"],
            )
            latent_mask = torch.ones(
                target_latent.size(0),
                target_latent.size(1),
                device=device,
                dtype=batch["future_mask"].dtype,
            )

        if is_train:
            optimizer.zero_grad()

        prefix_states = prefix_encoder(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
        )
        predicted_latent = predictor(
            prefix_states=prefix_states,
            prefix_mask=batch["prefix_mask"],
        )
        predictor_latent_loss = F.mse_loss(predicted_latent, target_latent)
        predictor_loss = predictor_latent_loss
        if train_config.predictor_decode_loss_weight > 0.0:
            decoded_logits = autoencoder.decode_latent(
                latent=predicted_latent,
                future_mask=batch["future_mask"],
            )
            decode_loss = masked_token_cross_entropy(
                logits=decoded_logits,
                target_ids=batch["future_ids"],
                mask=batch["future_mask"],
            )
            predictor_loss = predictor_loss + train_config.predictor_decode_loss_weight * decode_loss

        context_ids, context_mask = build_context_inputs(batch)
        context_states = context_encoder(
            context_ids=context_ids,
            context_mask=context_mask,
        )

        target_timesteps = noise_schedule.sample_timesteps(batch["future_ids"].size(0))
        noisy_target_latent, _ = noise_schedule.add_noise(target_latent, target_timesteps)
        target_self_condition = build_self_condition_latent(
            denoiser=denoiser,
            noisy_latent=noisy_target_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=target_timesteps,
            latent_mask=latent_mask,
        )
        predicted_target_clean = denoiser(
            noisy_latent=noisy_target_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=target_timesteps,
            future_mask=latent_mask,
            self_condition_latent=target_self_condition,
        )
        target_denoise_loss = masked_latent_mse(
            predicted_latent=predicted_target_clean,
            target_latent=target_latent,
            latent_mask=latent_mask,
        )

        predictor_timesteps = sample_timesteps_with_cap(
            noise_schedule=noise_schedule,
            batch_size=batch["future_ids"].size(0),
            max_timestep=train_config.predictor_refine_max_t,
        )
        noisy_predictor_latent, _ = noise_schedule.add_noise(predicted_latent, predictor_timesteps)
        predictor_self_condition = build_self_condition_latent(
            denoiser=denoiser,
            noisy_latent=noisy_predictor_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=predictor_timesteps,
            latent_mask=latent_mask,
        )
        predicted_refined_clean = denoiser(
            noisy_latent=noisy_predictor_latent,
            context_states=context_states,
            context_mask=context_mask,
            timesteps=predictor_timesteps,
            future_mask=latent_mask,
            self_condition_latent=predictor_self_condition,
        )
        predictor_refine_loss = masked_latent_mse(
            predicted_latent=predicted_refined_clean,
            target_latent=target_latent,
            latent_mask=latent_mask,
        )

        loss = (
            predictor_loss
            + train_config.target_denoise_loss_weight * target_denoise_loss
            + train_config.predictor_refine_loss_weight * predictor_refine_loss
        )

        if is_train:
            noise_schedule.update_with_losses(timesteps=target_timesteps, losses=((predicted_target_clean - target_latent) ** 2).mean(dim=(1, 2)))
            loss.backward()
            if train_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(prefix_encoder.parameters())
                    + list(predictor.parameters())
                    + list(context_encoder.parameters())
                    + list(denoiser.parameters()),
                    train_config.grad_clip_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_predictor_loss += predictor_loss.item()
        total_target_denoise_loss += target_denoise_loss.item()
        total_predictor_refine_loss += predictor_refine_loss.item()
        total_batches += 1

        if step % train_config.log_every == 0 or step == len(dataloader):
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred=f"{predictor_loss.item():.4f}",
                tgt=f"{target_denoise_loss.item():.4f}",
                init=f"{predictor_refine_loss.item():.4f}",
            )

    normalizer = max(total_batches, 1)
    return {
        "loss": total_loss / normalizer,
        "predictor_loss": total_predictor_loss / normalizer,
        "target_denoise_loss": total_target_denoise_loss / normalizer,
        "predictor_refine_loss": total_predictor_refine_loss / normalizer,
    }


def save_joint_checkpoint(
    prefix_encoder: PrefixEncoder,
    predictor: FutureLatentPredictor,
    context_encoder: ContextEncoder,
    denoiser: LatentDenoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "prefix_encoder_state_dict": prefix_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "context_encoder_state_dict": context_encoder.state_dict(),
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        checkpoint_path,
    )


def export_component_checkpoints(
    prefix_encoder: PrefixEncoder,
    predictor: FutureLatentPredictor,
    context_encoder: ContextEncoder,
    denoiser: LatentDenoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "prefix_encoder_state_dict": prefix_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        },
        checkpoint_dir / "predictor_best.pt",
    )
    torch.save(
        {
            "epoch": epoch,
            "context_encoder_state_dict": context_encoder.state_dict(),
            "denoiser_state_dict": denoiser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        },
        checkpoint_dir / "denoiser_best.pt",
    )


def append_epoch_log(
    log_path: Path,
    epoch: int,
    split: str,
    metrics: dict[str, float],
    device: torch.device,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": epoch, "split": split, "device": str(device), **metrics}
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/joint_predictor_denoiser_roberta.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = DataConfig.from_dict(config)
    train_config = JointTrainConfig.from_dict(config)
    denoiser_config = LatentDenoiserConfig.from_dict(config)

    device = resolve_device(train_config.device)
    print(f"Using device: {device}")
    _, train_loader, val_loader = build_dataloaders(data_config)

    autoencoder = load_autoencoder(config=config, checkpoint_path=train_config.ae_checkpoint_path, device=device)
    prefix_encoder, predictor = load_predictor_components(
        config=config,
        checkpoint_path=train_config.predictor_checkpoint_path,
        device=device,
    )
    context_encoder, denoiser = load_denoiser_components(
        config=config,
        checkpoint_path=train_config.denoiser_checkpoint_path,
        device=device,
    )

    trainable_parameters = [
        parameter
        for parameter in (
            list(prefix_encoder.parameters())
            + list(predictor.parameters())
            + list(context_encoder.parameters())
            + list(denoiser.parameters())
        )
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
    noise_schedule = DiffusionNoiseSchedule(
        num_steps=denoiser_config.num_diffusion_steps,
        schedule_type=config["model"].get("noise_schedule", "sqrt"),
        timestep_sampling=config["model"].get("timestep_sampling", "loss_aware"),
        device=device,
    )

    best_val_loss = float("inf")
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_path = Path(train_config.log_dir) / "joint_train.jsonl"
    if warmup_steps > 0:
        print(f"Using linear LR warmup for {warmup_steps} steps ({train_config.warmup_ratio:.1%} of training).")

    for epoch in range(1, train_config.num_epochs + 1):
        print(f"Epoch {epoch}/{train_config.num_epochs}")
        train_metrics = run_epoch(
            autoencoder=autoencoder,
            prefix_encoder=prefix_encoder,
            predictor=predictor,
            context_encoder=context_encoder,
            denoiser=denoiser,
            dataloader=train_loader,
            noise_schedule=noise_schedule,
            train_config=train_config,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                autoencoder=autoencoder,
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                context_encoder=context_encoder,
                denoiser=denoiser,
                dataloader=val_loader,
                noise_schedule=noise_schedule,
                train_config=train_config,
                device=device,
                optimizer=None,
                scheduler=None,
            )

        print(
            "train_loss: {loss:.4f} | pred: {predictor_loss:.4f} | target_denoise: {target_denoise_loss:.4f} | "
            "predictor_refine: {predictor_refine_loss:.4f}".format(**train_metrics)
        )
        print(
            "val_loss:   {loss:.4f} | pred: {predictor_loss:.4f} | target_denoise: {target_denoise_loss:.4f} | "
            "predictor_refine: {predictor_refine_loss:.4f}".format(**val_metrics)
        )
        append_epoch_log(log_path=log_path, epoch=epoch, split="train", metrics=train_metrics, device=device)
        append_epoch_log(log_path=log_path, epoch=epoch, split="val", metrics=val_metrics, device=device)

        if train_config.save_every_epoch:
            save_joint_checkpoint(
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                context_encoder=context_encoder,
                denoiser=denoiser,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_dir / f"joint_epoch_{epoch}.pt",
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_joint_checkpoint(
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                context_encoder=context_encoder,
                denoiser=denoiser,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                checkpoint_path=checkpoint_dir / "joint_best.pt",
            )
            export_component_checkpoints(
                prefix_encoder=prefix_encoder,
                predictor=predictor,
                context_encoder=context_encoder,
                denoiser=denoiser,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                checkpoint_dir=checkpoint_dir,
            )


if __name__ == "__main__":
    main()
