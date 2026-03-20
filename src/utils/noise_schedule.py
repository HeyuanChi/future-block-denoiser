from __future__ import annotations

import torch


class DiffusionNoiseSchedule:
    """Simple linear beta schedule for latent denoising experiments."""

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")

        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=self.device)

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clean_latent: [B, F, D]
            timesteps: [B]
        Returns:
            noisy_latent: [B, F, D]
            noise: [B, F, D]
        """
        noise = torch.randn_like(clean_latent)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        noisy_latent = torch.sqrt(alpha_bar) * clean_latent + torch.sqrt(1.0 - alpha_bar) * noise
        return noisy_latent, noise
