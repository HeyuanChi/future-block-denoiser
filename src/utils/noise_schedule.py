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
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_bars[:-1]],
            dim=0,
        )

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

    def add_noise_around_anchor(
        self,
        clean_latent: torch.Tensor,
        anchor_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Diffuses the residual between an anchor latent and the clean latent.

        Args:
            clean_latent: [B, F, D]
            anchor_latent: [B, F, D]
            timesteps: [B]
        Returns:
            noisy_latent: [B, F, D]
            noise: [B, F, D]
        """
        residual = clean_latent - anchor_latent
        noisy_residual, noise = self.add_noise(residual, timesteps)
        return anchor_latent + noisy_residual, noise

    def predict_clean_from_noise(
        self,
        noisy_latent: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return (noisy_latent - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

    def predict_clean_from_noise_around_anchor(
        self,
        noisy_latent: torch.Tensor,
        anchor_latent: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        residual = noisy_latent - anchor_latent
        clean_residual = (residual - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        return anchor_latent + clean_residual

    def step_ddpm_mean(
        self,
        noisy_latent: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic DDPM mean update without posterior sampling noise.

        Args:
            noisy_latent: [B, F, D]
            predicted_noise: [B, F, D]
            timesteps: [B]
        Returns:
            previous_latent: [B, F, D]
        """
        beta_t = self.betas[timesteps].view(-1, 1, 1)
        alpha_t = self.alphas[timesteps].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1)

        mean = (noisy_latent - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise) / torch.sqrt(alpha_t)

        is_not_final = (timesteps > 0).view(-1, 1, 1)
        previous_latent = torch.where(is_not_final, mean, self.predict_clean_from_noise(noisy_latent, predicted_noise, timesteps))
        return previous_latent

    def step_ddpm_mean_around_anchor(
        self,
        noisy_latent: torch.Tensor,
        anchor_latent: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        residual = noisy_latent - anchor_latent
        previous_residual = self.step_ddpm_mean(
            noisy_latent=residual,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )
        return anchor_latent + previous_residual
