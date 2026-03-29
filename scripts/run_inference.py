from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DataConfig, build_dataloaders
from src.models.future_autoencoder import FutureAutoencoderConfig
from src.models.latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_denoiser import load_autoencoder
from src.utils.noise_schedule import DiffusionNoiseSchedule


def load_denoiser_components(
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, PrefixEncoder, LatentDenoiser]:
    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=config["training"]["ae_checkpoint_path"],
        device=device,
    )

    prefix_encoder = PrefixEncoder(PrefixEncoderConfig.from_dict(config)).to(device)
    denoiser = LatentDenoiser(LatentDenoiserConfig.from_dict(config)).to(device)

    checkpoint = torch.load(
        config["training"]["denoiser_checkpoint_path"],
        map_location=device,
    )
    prefix_encoder.load_state_dict(checkpoint["prefix_encoder_state_dict"])
    denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    prefix_encoder.eval()
    denoiser.eval()
    return autoencoder, prefix_encoder, denoiser


def iterative_refine_latent(
    denoiser: LatentDenoiser,
    noise_schedule: DiffusionNoiseSchedule,
    prefix_states: torch.Tensor,
    prefix_mask: torch.Tensor,
    future_mask: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """
    Runs a deterministic reverse diffusion loop from Gaussian noise.
    """
    batch_size = prefix_states.size(0)
    latent_len = denoiser.latent_len
    latent_dim = denoiser.config.latent_dim
    if denoiser.config.use_initializer:
        latent = denoiser.initialize_latent(
            prefix_states=prefix_states,
            prefix_mask=prefix_mask,
        )
    else:
        latent = torch.randn(batch_size, latent_len, latent_dim, device=prefix_states.device)
    latent_mask = torch.ones(batch_size, latent_len, device=prefix_states.device, dtype=prefix_mask.dtype)

    for timestep in reversed(range(num_steps)):
        timestep_tensor = torch.full((batch_size,), timestep, device=prefix_states.device, dtype=torch.long)
        predicted_noise = denoiser(
            noisy_latent=latent,
            prefix_states=prefix_states,
            timesteps=timestep_tensor,
            prefix_mask=prefix_mask,
            future_mask=latent_mask,
        )
        latent = noise_schedule.step_ddpm_mean(
            noisy_latent=latent,
            predicted_noise=predicted_noise,
            timesteps=timestep_tensor,
        )

    return latent


def decode_ids(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/denoiser.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"].setdefault("ae_checkpoint_path", "outputs/checkpoints/ae_best.pt")
    config["training"].setdefault("denoiser_checkpoint_path", "outputs/checkpoints/denoiser_best.pt")

    data_config = DataConfig.from_dict(config)
    data_config.batch_size = 1
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")

    tokenizer, _, val_loader = build_dataloaders(data_config)
    autoencoder, prefix_encoder, denoiser = load_denoiser_components(config, device)

    denoiser_config = LatentDenoiserConfig.from_dict(config)
    num_steps = args.num_steps or denoiser_config.num_diffusion_steps
    noise_schedule = DiffusionNoiseSchedule(
        num_steps=denoiser_config.num_diffusion_steps,
        device=device,
    )

    selected_batch = None
    for batch_index, batch in enumerate(val_loader):
        if batch_index == args.sample_index:
            selected_batch = batch
            break

    if selected_batch is None:
        raise ValueError(f"sample_index {args.sample_index} is out of range for the validation loader.")

    batch = move_batch_to_device(selected_batch, device)

    with torch.no_grad():
        target_latent, ae_logits = autoencoder(
            future_ids=batch["future_ids"],
            future_mask=batch["future_mask"],
        )
        ae_prediction_ids = ae_logits.argmax(dim=-1)

        prefix_states = prefix_encoder(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
        )
        denoised_latent = iterative_refine_latent(
            denoiser=denoiser,
            noise_schedule=noise_schedule,
            prefix_states=prefix_states,
            prefix_mask=batch["prefix_mask"],
            future_mask=batch["future_mask"],
            num_steps=num_steps,
        )
        denoised_logits = autoencoder.decode_latent(
            latent=denoised_latent,
            future_mask=batch["future_mask"],
        )
        denoised_prediction_ids = denoised_logits.argmax(dim=-1)

        oracle_timestep = torch.full(
            (batch["future_ids"].size(0),),
            denoiser_config.num_diffusion_steps - 1,
            device=device,
            dtype=torch.long,
        )
        oracle_noisy_latent, _ = noise_schedule.add_noise(target_latent, oracle_timestep)
        oracle_predicted_noise = denoiser(
            noisy_latent=oracle_noisy_latent,
            prefix_states=prefix_states,
            timesteps=oracle_timestep,
            prefix_mask=batch["prefix_mask"],
            future_mask=torch.ones(
                target_latent.size(0),
                target_latent.size(1),
                device=device,
                dtype=batch["future_mask"].dtype,
            ),
        )
        oracle_latent = noise_schedule.predict_clean_from_noise(
            noisy_latent=oracle_noisy_latent,
            predicted_noise=oracle_predicted_noise,
            timesteps=oracle_timestep,
        )
        oracle_logits = autoencoder.decode_latent(
            latent=oracle_latent,
            future_mask=batch["future_mask"],
        )
        oracle_prediction_ids = oracle_logits.argmax(dim=-1)

        latent_mse = torch.mean((denoised_latent - target_latent) ** 2).item()
        oracle_latent_mse = torch.mean((oracle_latent - target_latent) ** 2).item()

    prefix_text = decode_ids(tokenizer, batch["prefix_ids"][0].cpu())
    future_text = decode_ids(tokenizer, batch["future_ids"][0].cpu())
    ae_text = decode_ids(tokenizer, ae_prediction_ids[0].cpu())
    denoised_text = decode_ids(tokenizer, denoised_prediction_ids[0].cpu())
    oracle_text = decode_ids(tokenizer, oracle_prediction_ids[0].cpu())

    print("\nPrefix:")
    print(prefix_text)
    print("\nGround Truth Future:")
    print(future_text)
    print("\nAE Reconstruction:")
    print(ae_text)
    print("\nDenoised Prediction:")
    print(denoised_text)
    print("\nOracle Denoise From True Latent + Noise:")
    print(oracle_text)
    print(f"\nLatent MSE to AE target: {latent_mse:.4f}")
    print(f"Oracle latent MSE to AE target: {oracle_latent_mse:.4f}")


if __name__ == "__main__":
    main()
