"""Utilities for accumulating Gaussian statistics over augmented windows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch


class GaussianStatsCollector:
    """
    Collect mean/variance/covariance for column vectors produced from
    augmented windows.

    Each augmented window has shape [B, C, T]. For statistics we treat every
    time step as one vertical vector of length C (channels/features) and stack
    them over the whole dataset → shape [N, C].
    """

    def __init__(self, n_features: int, device: Union[str, torch.device] = "cpu"):
        self.n_features = n_features
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Reset all running accumulators."""
        self.count: int = 0
        self.sum = torch.zeros(self.n_features, dtype=torch.float64, device=self.device)
        self.sum_sq = torch.zeros_like(self.sum)
        self.sum_outer = torch.zeros(
            (self.n_features, self.n_features), dtype=torch.float64, device=self.device
        )

    @staticmethod
    def to_column_vectors(batch: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of shape [B, C, T] to stacked column vectors [B*T, C].

        We treat each time step as one vertical vector of length C (channels),
        then stack all time steps across the batch.
        """

        if batch.ndim != 3:
            raise ValueError(
                f"Expected tensor with 3 dims [B, C, T], got {batch.shape}"
            )

        # [B, C, T] → [B, T, C] → [B*T, C]
        return batch.permute(0, 2, 1).reshape(-1, batch.shape[1]).to(torch.float64)

    def update(self, vectors: torch.Tensor) -> None:
        """Update running statistics with vectors of shape [N, C]."""
        if vectors.numel() == 0:
            return

        vectors = vectors.to(self.device, dtype=torch.float64)
        batch_count = vectors.shape[0]

        self.count += batch_count
        self.sum += vectors.sum(dim=0)
        self.sum_sq += (vectors * vectors).sum(dim=0)
        self.sum_outer += vectors.T @ vectors

    def update_from_augmented(self, augmented_batch: torch.Tensor) -> None:
        """Helper to update directly from an augmented batch [B, C, T]."""
        vectors = self.to_column_vectors(augmented_batch)
        self.update(vectors)

    def finalize(self) -> Dict[str, torch.Tensor]:
        """Return mean, variance, covariance (all torch.float64)."""
        if self.count == 0:
            raise RuntimeError(
                "No samples have been accumulated; cannot finalize stats."
            )

        mean = self.sum / self.count
        variance = torch.clamp(self.sum_sq / self.count - mean * mean, min=0.0)

        if self.count > 1:
            cov = (self.sum_outer - self.count * torch.outer(mean, mean)) / (
                self.count - 1
            )
        else:
            cov = torch.zeros_like(self.sum_outer)

        return {
            "mean": mean,
            "variance": variance,
            "covariance": cov,
            "count": torch.tensor(self.count, dtype=torch.long, device=self.device),
        }

    def save(
        self, path: Union[str, Path], stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> Path:
        """Finalize (if needed) and persist stats to disk as a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = stats if stats is not None else self.finalize()
        torch.save(payload, path)
        return path
