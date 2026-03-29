from __future__ import annotations

from collections import deque

import torch


class DiffusionNoiseSchedule:
    """Diffusion schedule with text-friendly defaults and loss-aware timestep sampling."""

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "sqrt",
        timestep_sampling: str = "loss_aware",
        loss_history_per_step: int = 32,
        eps: float = 1e-6,
        device: torch.device | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")
        self.timestep_sampling = timestep_sampling
        self.eps = eps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)
        elif schedule_type == "sqrt":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, device=self.device) ** 2
        else:
            raise ValueError(f"Unsupported schedule_type={schedule_type!r}.")
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alpha_bars[:-1]],
            dim=0,
        )
        self.loss_history = [deque(maxlen=loss_history_per_step) for _ in range(num_steps)]

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        if self.timestep_sampling == "uniform":
            return torch.randint(0, self.num_steps, (batch_size,), device=self.device)

        if self.timestep_sampling != "loss_aware":
            raise ValueError(f"Unsupported timestep_sampling={self.timestep_sampling!r}.")

        weights = []
        for history in self.loss_history:
            if not history:
                weights.append(1.0)
            else:
                mean_loss = sum(history) / len(history)
                weights.append((mean_loss + self.eps) ** 0.5)
        probabilities = torch.tensor(weights, device=self.device, dtype=torch.float)
        probabilities = probabilities / probabilities.sum()
        return torch.multinomial(probabilities, batch_size, replacement=True)

    def update_with_losses(
        self,
        timesteps: torch.Tensor,
        losses: torch.Tensor,
    ) -> None:
        for timestep, loss in zip(timesteps.detach().cpu().tolist(), losses.detach().cpu().tolist(), strict=False):
            self.loss_history[timestep].append(float(loss))

    def add_noise(
        self,
        clean_latent: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clean_latent: [B, F, D]
            timesteps: [B]
        Returns:
            noisy_latent: [B, F, D]
            noise: [B, F, D]
        """
        if noise is None:
            noise = torch.randn_like(clean_latent)
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        noisy_latent = torch.sqrt(alpha_bar) * clean_latent + torch.sqrt(1.0 - alpha_bar) * noise
        return noisy_latent, noise

    def predict_clean_from_noise(
        self,
        noisy_latent: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return (noisy_latent - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

    def predict_noise_from_clean(
        self,
        noisy_latent: torch.Tensor,
        predicted_clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return (noisy_latent - torch.sqrt(alpha_bar) * predicted_clean) / torch.sqrt(1.0 - alpha_bar)

    def step_ddpm_mean_from_clean(
        self,
        noisy_latent: torch.Tensor,
        predicted_clean: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        predicted_noise = self.predict_noise_from_clean(
            noisy_latent=noisy_latent,
            predicted_clean=predicted_clean,
            timesteps=timesteps,
        )
        return self.step_ddpm_mean(
            noisy_latent=noisy_latent,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

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
        previous_latent = torch.where(
            is_not_final,
            mean,
            self.predict_clean_from_noise(noisy_latent, predicted_noise, timesteps),
        )
        return previous_latent
