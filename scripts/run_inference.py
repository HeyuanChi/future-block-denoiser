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
from src.models.future_latent_initializer import FutureLatentInitializer
from src.models.future_latent_initializer import FutureLatentInitializerConfig
from src.models.latent_denoiser import LatentDenoiser, LatentDenoiserConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_denoiser import load_autoencoder
from src.utils.noise_schedule import DiffusionNoiseSchedule


def load_denoiser_components(
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, PrefixEncoder, FutureLatentInitializer | None, LatentDenoiser, bool]:
    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=config["training"]["ae_checkpoint_path"],
        device=device,
    )

    prefix_encoder = PrefixEncoder(PrefixEncoderConfig.from_dict(config)).to(device)
    use_coarse_initializer = bool(config["training"].get("use_coarse_initializer", False))
    initializer = None
    if use_coarse_initializer:
        initializer = FutureLatentInitializer(FutureLatentInitializerConfig.from_dict(config)).to(device)
    denoiser = LatentDenoiser(LatentDenoiserConfig.from_dict(config)).to(device)

    checkpoint = torch.load(
        config["training"]["denoiser_checkpoint_path"],
        map_location=device,
    )
    prefix_encoder.load_state_dict(checkpoint["prefix_encoder_state_dict"])
    if initializer is not None:
        initializer_loaded = False
        if "initializer_state_dict" in checkpoint:
            initializer.load_state_dict(checkpoint["initializer_state_dict"])
            initializer_loaded = True
        elif config["training"].get("initializer_checkpoint_path"):
            initializer_checkpoint = torch.load(config["training"]["initializer_checkpoint_path"], map_location=device)
            initializer.load_state_dict(initializer_checkpoint["initializer_state_dict"])
            initializer_loaded = True
        if not initializer_loaded:
            raise ValueError(
                "use_coarse_initializer=True, but no initializer weights were found in the denoiser "
                "checkpoint or at training.initializer_checkpoint_path."
            )
    denoiser.load_state_dict(checkpoint["denoiser_state_dict"])

    prefix_encoder.eval()
    if initializer is not None:
        initializer.eval()
    denoiser.eval()
    return autoencoder, prefix_encoder, initializer, denoiser, use_coarse_initializer


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
    future_len = future_mask.size(1)
    latent_dim = denoiser.config.latent_dim
    latent = torch.randn(batch_size, future_len, latent_dim, device=prefix_states.device)

    for timestep in reversed(range(num_steps)):
        timestep_tensor = torch.full((batch_size,), timestep, device=prefix_states.device, dtype=torch.long)
        predicted_noise = denoiser(
            noisy_latent=latent,
            prefix_states=prefix_states,
            timesteps=timestep_tensor,
            prefix_mask=prefix_mask,
            future_mask=future_mask,
        )
        latent = noise_schedule.step_ddpm_mean(
            noisy_latent=latent,
            predicted_noise=predicted_noise,
            timesteps=timestep_tensor,
        )

    return latent


def iterative_refine_from_coarse_latent(
    denoiser: LatentDenoiser,
    noise_schedule: DiffusionNoiseSchedule,
    coarse_latent: torch.Tensor,
    prefix_states: torch.Tensor,
    prefix_mask: torch.Tensor,
    future_mask: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    batch_size = prefix_states.size(0)
    latent = coarse_latent + torch.randn_like(coarse_latent)

    for timestep in reversed(range(num_steps)):
        timestep_tensor = torch.full((batch_size,), timestep, device=prefix_states.device, dtype=torch.long)
        predicted_noise = denoiser(
            noisy_latent=latent,
            prefix_states=prefix_states,
            timesteps=timestep_tensor,
            prefix_mask=prefix_mask,
            future_mask=future_mask,
        )
        latent = noise_schedule.step_ddpm_mean_around_anchor(
            noisy_latent=latent,
            anchor_latent=coarse_latent,
            predicted_noise=predicted_noise,
            timesteps=timestep_tensor,
        )

    return latent


def direct_residual_refine_from_coarse_latent(
    denoiser: LatentDenoiser,
    coarse_latent: torch.Tensor,
    prefix_states: torch.Tensor,
    prefix_mask: torch.Tensor,
    future_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = prefix_states.size(0)
    timestep_tensor = torch.zeros((batch_size,), device=prefix_states.device, dtype=torch.long)
    predicted_delta = denoiser(
        noisy_latent=coarse_latent,
        prefix_states=prefix_states,
        timesteps=timestep_tensor,
        prefix_mask=prefix_mask,
        future_mask=future_mask,
    )
    return coarse_latent + predicted_delta


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
    config["training"].setdefault("initializer_checkpoint_path", "outputs/checkpoints/initializer_best.pt")
    config["training"].setdefault("coarse_refinement_mode", "noise")

    data_config = DataConfig.from_dict(config)
    data_config.batch_size = 1
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")

    tokenizer, _, val_loader = build_dataloaders(data_config)
    autoencoder, prefix_encoder, initializer, denoiser, use_coarse_initializer = load_denoiser_components(config, device)

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
        coarse_prediction_ids = None
        coarse_latent = None
        if use_coarse_initializer:
            if initializer is None:
                raise ValueError("Configured to use a coarse initializer, but no initializer weights were loaded.")
            coarse_latent = initializer(
                prefix_states=prefix_states,
                prefix_mask=batch["prefix_mask"],
                future_mask=batch["future_mask"],
            )
            coarse_logits = autoencoder.decode_latent(
                latent=coarse_latent,
                future_mask=batch["future_mask"],
            )
            coarse_prediction_ids = coarse_logits.argmax(dim=-1)
            if config["training"].get("coarse_refinement_mode", "noise") == "residual":
                denoised_latent = direct_residual_refine_from_coarse_latent(
                    denoiser=denoiser,
                    coarse_latent=coarse_latent,
                    prefix_states=prefix_states,
                    prefix_mask=batch["prefix_mask"],
                    future_mask=batch["future_mask"],
                )
            else:
                denoised_latent = iterative_refine_from_coarse_latent(
                    denoiser=denoiser,
                    noise_schedule=noise_schedule,
                    coarse_latent=coarse_latent,
                    prefix_states=prefix_states,
                    prefix_mask=batch["prefix_mask"],
                    future_mask=batch["future_mask"],
                    num_steps=num_steps,
                )
        else:
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

        if use_coarse_initializer and config["training"].get("coarse_refinement_mode", "noise") == "residual":
            oracle_latent = target_latent
        else:
            oracle_timestep = torch.full(
                (batch["future_ids"].size(0),),
                denoiser_config.num_diffusion_steps - 1,
                device=device,
                dtype=torch.long,
            )
            if use_coarse_initializer:
                if coarse_latent is None:
                    raise ValueError("coarse_latent must be available in coarse initializer mode.")
                oracle_noisy_latent, _ = noise_schedule.add_noise_around_anchor(
                    clean_latent=target_latent,
                    anchor_latent=coarse_latent,
                    timesteps=oracle_timestep,
                )
            else:
                oracle_noisy_latent, _ = noise_schedule.add_noise(target_latent, oracle_timestep)
            oracle_predicted_noise = denoiser(
                noisy_latent=oracle_noisy_latent,
                prefix_states=prefix_states,
                timesteps=oracle_timestep,
                prefix_mask=batch["prefix_mask"],
                future_mask=batch["future_mask"],
            )
            if use_coarse_initializer:
                oracle_latent = noise_schedule.predict_clean_from_noise_around_anchor(
                    noisy_latent=oracle_noisy_latent,
                    anchor_latent=coarse_latent,
                    predicted_noise=oracle_predicted_noise,
                    timesteps=oracle_timestep,
                )
            else:
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
        coarse_latent_mse = None if coarse_latent is None else torch.mean((coarse_latent - target_latent) ** 2).item()

    prefix_text = decode_ids(tokenizer, batch["prefix_ids"][0].cpu())
    future_text = decode_ids(tokenizer, batch["future_ids"][0].cpu())
    ae_text = decode_ids(tokenizer, ae_prediction_ids[0].cpu())
    coarse_text = None if coarse_prediction_ids is None else decode_ids(tokenizer, coarse_prediction_ids[0].cpu())
    denoised_text = decode_ids(tokenizer, denoised_prediction_ids[0].cpu())
    oracle_text = decode_ids(tokenizer, oracle_prediction_ids[0].cpu())

    print("\nPrefix:")
    print(prefix_text)
    print("\nGround Truth Future:")
    print(future_text)
    print("\nAE Reconstruction:")
    print(ae_text)
    if coarse_text is not None:
        print("\nCoarse Initializer Prediction:")
        print(coarse_text)
    print("\nDenoised Prediction:")
    print(denoised_text)
    oracle_label = "Oracle Denoise From True Latent + Noise"
    if use_coarse_initializer and config["training"].get("coarse_refinement_mode", "noise") == "residual":
        oracle_label = "Oracle Target Latent Decode"
    print(f"\n{oracle_label}:")
    print(oracle_text)
    if coarse_latent_mse is not None:
        print(f"\nCoarse latent MSE to AE target: {coarse_latent_mse:.4f}")
    print(f"\nLatent MSE to AE target: {latent_mse:.4f}")
    print(f"Oracle latent MSE to AE target: {oracle_latent_mse:.4f}")


if __name__ == "__main__":
    main()
