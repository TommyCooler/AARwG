import torch
import torch.nn as nn


class RandomTimeMasking(nn.Module):

    def __init__(self, mask_ratio: float = 0.15):
        super().__init__()

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}"
            )

        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or self.mask_ratio == 0.0:
            return x

        B, C, T = x.shape

        n_mask = int(T * self.mask_ratio)
        if n_mask <= 0:
            return x

        # chọn ngẫu nhiên các time-step để mask, dùng chung cho cả batch & channel
        mask_indices = torch.randperm(T, device=x.device)[:n_mask]  # [n_mask]

        # mask theo time-step: True = keep, False = mask
        time_mask = torch.ones(T, device=x.device, dtype=torch.bool)
        time_mask[mask_indices] = False  # [T]
        time_mask = time_mask.view(1, 1, T)  # [1,1,T] → broadcast

        masked_x = x * time_mask  # mask = 0
        return masked_x


class InferenceMasking(nn.Module):

    def __init__(self, mask_ratio: float = 0.15):
        super().__init__()

        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}"
            )

        self.mask_ratio = mask_ratio
        self.window_index = 0  # Track current window index

    def reset(self):
        self.window_index = 0

    def forward(self, x: torch.Tensor, window_idx: int = None) -> torch.Tensor:

        if self.mask_ratio == 0.0:
            return x

        batch_size, _, seq_len = x.shape

        # Determine window index
        if window_idx is not None:
            current_window_idx = window_idx
        else:
            current_window_idx = self.window_index
            self.window_index += batch_size

        # Create mask based on window index
        if current_window_idx == 0:
            # Window 0: Random time masking (temporal)
            n_mask = int(seq_len * self.mask_ratio)
            if n_mask > 0:
                mask_indices = torch.randperm(seq_len, device=x.device)[:n_mask]
                mask = torch.ones_like(x, dtype=torch.bool)
                mask[:, :, mask_indices] = False
            else:
                mask = torch.ones_like(x, dtype=torch.bool)
        else:
            # Window 1 to last: Only mask the last time-step
            mask = torch.ones_like(x, dtype=torch.bool)
            mask[:, :, -1] = False  # Mask last time-step for all channels

        # Apply zero masking
        masked_x = x * mask

        return masked_x
