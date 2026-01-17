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

    @staticmethod
    def compute_negative_log_likelihood(
        vectors: torch.Tensor,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood of vectors under single Gaussian (k=1).

        Formula (4) with k=1, no mixture weights:
        ℓ(x) = -log N(x | μ, Σ)
             = 0.5 * d * log(2π) + 0.5 * log|Σ| + 0.5 * (x-μ)^T Σ^(-1) (x-μ)

        Args:
            vectors: [N, C] feature vectors
            mean: [C] mean vector
            covariance: [C, C] covariance matrix
            eps: regularization for numerical stability

        Returns:
            [N] negative log-likelihood for each vector
        """
        device = vectors.device
        dtype = vectors.dtype

        diff = vectors - mean.to(dtype).unsqueeze(0)  # [N, C]
        dim = vectors.shape[1]

        # Add regularization to diagonal for stability
        cov_reg = covariance.to(dtype) + eps * torch.eye(
            dim, device=device, dtype=dtype
        )

        # Cholesky decomposition for log determinant
        try:
            L = torch.linalg.cholesky(cov_reg)
            log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        except RuntimeError:
            # Fallback to eigenvalue decomposition
            eigvals = torch.linalg.eigvalsh(cov_reg)
            eigvals = torch.clamp(eigvals, min=eps)
            log_det = torch.sum(torch.log(eigvals))

        # Compute (x-μ)^T Σ^(-1) (x-μ)
        try:
            cov_inv = torch.linalg.inv(cov_reg)
        except RuntimeError:
            cov_inv = torch.linalg.pinv(cov_reg)

        mahal_dist = torch.sum(diff @ cov_inv * diff, dim=1)  # [N]

        # ℓ(x) = 0.5 * d * log(2π) + 0.5 * log|Σ| + 0.5 * mahal_dist
        const_term = (
            0.5
            * dim
            * torch.log(torch.tensor(2.0 * 3.14159265359, device=device, dtype=dtype))
        )
        nll = const_term + 0.5 * log_det + 0.5 * mahal_dist

        return nll

    @staticmethod
    def anomaly_score_with_sigmoid(
        nll: torch.Tensor,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> torch.Tensor:
        """
        Convert negative log-likelihood to anomaly score [0, 1] using sigmoid.

        Formula (5):
        λ = 0.5 / (1 + exp(c * [ℓ + x₀]))

        where c is scale and x₀ is shift.

        Args:
            nll: [N] negative log-likelihood values
            scale: c parameter (scale)
            shift: x₀ parameter (shift)

        Returns:
            [N] anomaly scores in [0, 1]
        """
        sigmoid_input = scale * (nll + shift)
        lambda_v = 0.5 / (1.0 + torch.exp(sigmoid_input))
        return lambda_v

    def save(
        self, path: Union[str, Path], stats: Optional[Dict[str, torch.Tensor]] = None
    ) -> Path:
        """Finalize (if needed) and persist stats to disk as a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = stats if stats is not None else self.finalize()
        torch.save(payload, path)
        return path
