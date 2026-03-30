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
from src.models.future_latent_predictor import FutureLatentPredictor, FutureLatentPredictorConfig
from src.models.prefix_encoder import PrefixEncoder, PrefixEncoderConfig
from src.training.train_ae import load_config, move_batch_to_device, resolve_device
from src.training.train_four_slot_predictor import load_autoencoder


def decode_ids(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def load_components(
    config: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, PrefixEncoder, FutureLatentPredictor]:
    autoencoder = load_autoencoder(
        config=config,
        checkpoint_path=config["training"]["ae_checkpoint_path"],
        device=device,
    )
    prefix_encoder = PrefixEncoder(PrefixEncoderConfig.from_dict(config)).to(device)
    predictor = FutureLatentPredictor(FutureLatentPredictorConfig.from_dict(config)).to(device)

    checkpoint = torch.load(
        config["training"]["predictor_checkpoint_path"],
        map_location=device,
    )
    prefix_encoder.load_state_dict(checkpoint["prefix_encoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])

    prefix_encoder.eval()
    predictor.eval()
    return autoencoder, prefix_encoder, predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/four_slot_predictor_roberta.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config["training"].setdefault(
        "ae_checkpoint_path",
        "outputs/checkpoints/ae_four_slot_roberta_stable/ae_best.pt",
    )
    config["training"].setdefault(
        "predictor_checkpoint_path",
        "outputs/checkpoints/four_slot_predictor_roberta/predictor_best.pt",
    )

    data_config = DataConfig.from_dict(config)
    data_config.batch_size = 1
    device = resolve_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")

    tokenizer, _, val_loader = build_dataloaders(data_config)
    autoencoder, prefix_encoder, predictor = load_components(config, device)

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
        if autoencoder.config.backbone_type == "bart":
            ae_prediction_ids = autoencoder.generate_from_latent(target_latent)
        else:
            ae_prediction_ids = ae_logits.argmax(dim=-1)

        prefix_states = prefix_encoder(
            prefix_ids=batch["prefix_ids"],
            prefix_mask=batch["prefix_mask"],
        )
        predicted_latent = predictor(
            prefix_states=prefix_states,
            prefix_mask=batch["prefix_mask"],
        )
        if autoencoder.config.backbone_type == "bart":
            predicted_ids = autoencoder.generate_from_latent(predicted_latent)
        else:
            predicted_logits = autoencoder.decode_latent(
                latent=predicted_latent,
                future_mask=batch["future_mask"],
            )
            predicted_ids = predicted_logits.argmax(dim=-1)

        latent_mse = torch.mean((predicted_latent - target_latent) ** 2).item()

    prefix_text = decode_ids(tokenizer, batch["prefix_ids"][0].cpu())
    future_text = decode_ids(tokenizer, batch["future_ids"][0].cpu())
    ae_text = decode_ids(tokenizer, ae_prediction_ids[0].cpu())
    predicted_text = decode_ids(tokenizer, predicted_ids[0].cpu())

    print("\nPrefix:")
    print(prefix_text)
    print("\nGround Truth Future:")
    print(future_text)
    print("\nAE Reconstruction:")
    print(ae_text)
    print("\nPredicted Future:")
    print(predicted_text)
    print(f"\nPredicted 4-slot latent MSE to AE target: {latent_mse:.4f}")


if __name__ == "__main__":
    main()
