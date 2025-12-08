"""
Phase 2 Inference: Anomaly Detection với Best Threshold Search

Usage:
    python phase2/inference.py --checkpoint checkpoints/phase2_ucr_135_best.pt

hoặc tùy chỉnh:
    python phase2/inference.py --checkpoint <path> --dataset ucr --subset 135
"""

import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.phase2_dataloader import prepare_phase2_data, Phase2Dataset
from data.dataloader import (
    ucr_sub_ds_processing,
    smd_sub_ds_processing,
    smap_msl_sub_ds_processing,
    psm_sub_ds_processing,
    pd_sub_ds_processing,
    ecg_sub_ds_processing,
    gesture_sub_ds_processing,
)
from modules.augmentation import Augmentation
from modules.random_masking import InferenceMasking
from phase2.agf_tcn import Agf_TCN
from utils.point_adjustment import evaluate_with_pa


def map_window_scores_to_timeseries(
    timestep_scores_all_windows: np.ndarray,
    n_time_steps: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """
    Map window-based per-time-step scores back to original time series.

    Strategy:
    - Window 0: map all time-step scores to [0:window_size]
    - Window i (i>0): map only the LAST time-step score (at index window_size-1)
                      to time step (start_idx + window_size - 1)

    Args:
        timestep_scores_all_windows: (n_windows, window_size) - per-time-step scores for each window
        n_time_steps: Number of time steps in original time series
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        timeseries_scores: (n_time_steps,) - anomaly scores for each time step
    """
    n_windows = timestep_scores_all_windows.shape[0]

    # Initialize arrays
    timeseries_scores = np.full(n_time_steps, np.nan, dtype=np.float32)

    # Map window 0: all time steps
    window_0_scores = timestep_scores_all_windows[0]  # Shape: (window_size,)
    timeseries_scores[0:window_size] = window_0_scores

    # Map subsequent windows: only last time step
    for i in range(1, n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        last_time_step = end_idx - 1  # Last time step in this window

        # Only map to last time step if it's within bounds
        if last_time_step < n_time_steps:
            # Extract the last time-step score from this window
            last_timestep_score = timestep_scores_all_windows[i, -1]

            # Map score (no overlap because we only take the last time-step)
            timeseries_scores[last_time_step] = last_timestep_score

    # Fill NaN values (if any) with forward fill
    # This handles edge cases where some time steps might not have scores
    valid_mask = ~np.isnan(timeseries_scores)
    if not valid_mask.all():
        # Forward fill from last valid value
        last_valid_idx = -1
        for i in range(n_time_steps):
            if valid_mask[i]:
                last_valid_idx = i
            elif last_valid_idx >= 0:
                timeseries_scores[i] = timeseries_scores[last_valid_idx]

    return timeseries_scores


class Phase2Inference:
    """
    Inference cho Phase 2 Anomaly Detection
    """

    def __init__(
        self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"\n📦 Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config
        if "config" not in checkpoint:
            raise ValueError("Checkpoint thiếu 'config'! Không thể tái tạo model.")

        self.config = checkpoint["config"]
        print(f"  ✓ Config loaded")
        print(f"    Dataset: {self.config['dataset_name']}_{self.config['subset']}")
        print(f"    Window size: {self.config['window_size']}")
        print(f"    Channels: {self.config['n_channels']}")

        # Recreate Augmentation
        self.augmentation = Augmentation(
            in_channels=self.config["n_channels"],
            seq_len=self.config["window_size"],
            kernel_size=self.config["aug_kernel_size_cnn"],  
            num_layers=self.config["aug_num_layers"],
            dropout=self.config["aug_dropout"],
            temperature=self.config["aug_temperature"],
            hard=self.config["aug_hard_gumbel_softmax"],    
            transformer_d_model=self.config["aug_transformer_d_model"],
            transformer_nhead=self.config["aug_transformer_nhead"],
        )
        self.augmentation.load_state_dict(checkpoint["augmentation_state_dict"])
        self.augmentation.to(device)
        self.augmentation.eval()  # Disable dropout and set batchnorm to eval mode
        print(f"  ✓ Augmentation loaded")

        # Recreate AGF-TCN
        self.agf_tcn = Agf_TCN(
            num_inputs=self.config["n_channels"],
            num_channels=self.config["agf_tcn_channels"],
            dropout=self.config["dropout"],
            activation=self.config["activation"],
            fuse_type=self.config["fuse_type"],
            window_size=self.config["window_size"],
        )
        self.agf_tcn.load_state_dict(checkpoint["agf_tcn_state_dict"])
        self.agf_tcn.to(device)
        self.agf_tcn.eval()
        print(f"  ✓ AGF-TCN loaded")

        # Initialize inference masking
        self.inference_masking = InferenceMasking(
            mask_ratio=self.config["mask_ratio"],
        ).to(device)
        print(f"  ✓ Inference masking initialized")
        print(f"    Mask ratio: {self.config['mask_ratio']}")

        # Print checkpoint info
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            print(f"\n📊 Checkpoint metrics (from training):")
            print(f"  Train Loss: {metrics['train_loss']:.6f}")

    def predict(
        self, test_loader, stride=None, labels=None
    ):

        print("\n🔮 Running inference...")
        print("  Applying inference masking:")
        print("    - Window 0: Random time masking")
        print("    - Window 1 to last: Only mask last time-step")

        all_timestep_scores = []

        # Reset window index for masking
        self.inference_masking.reset()
        global_window_idx = 0

        with torch.no_grad():
            for batch_data, _ in tqdm(test_loader, desc="Computing anomaly scores"):
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.shape[0]

                # Apply inference masking with window index tracking
                masked_data = batch_data.clone()
                for i in range(batch_size):
                    window_data = batch_data[i : i + 1]  # (1, channels, seq_len)
                    masked_window = self.inference_masking(
                        window_data, window_idx=global_window_idx
                    )
                    masked_data[i] = masked_window[0]
                    global_window_idx += 1

                # Apply augmentation to masked data
                augmented_data = self.augmentation(masked_data)
                reconstructed = self.agf_tcn(augmented_data)
                timestep_losses = torch.mean(
                    (reconstructed - augmented_data) ** 2, dim=1
                )

                all_timestep_scores.append(timestep_losses.cpu().numpy())

        timestep_scores_all_windows = np.concatenate(all_timestep_scores, axis=0)

        print(
            f"  ✓ Computed anomaly scores for {len(timestep_scores_all_windows)} windows"
        )
        print(
            f"   Each window has {timestep_scores_all_windows.shape[1]} time-step scores"
        )

        stride = stride if stride is not None else self.config["stride"]
        window_size = self.config["window_size"]

        print(f"\n🔄 Mapping window scores to time series...")
        print(f"   Window size: {window_size}, Stride: {stride}")
        print(
            f"   Strategy: Window 0 → all time-steps, Window i>0 → last time-step only"
        )

        anomaly_scores = map_window_scores_to_timeseries(
            timestep_scores_all_windows=timestep_scores_all_windows,
            n_time_steps=len(labels),
            window_size=window_size,
            stride=stride,
        )

        print(f"  ✓ Mapped to {len(anomaly_scores)} time steps")
        print(
            f"   Coverage: {np.sum(~np.isnan(anomaly_scores))}/{len(anomaly_scores)} steps have scores"
        )

        # Evaluate with Point Adjustment (always searches for best threshold)
        print("\n🔍 Searching for best threshold (with Point Adjustment)...")

        metrics = evaluate_with_pa(anomaly_scores=anomaly_scores, labels=labels)

        return metrics, anomaly_scores, labels

    def print_results(self, metrics):
        """In kết quả inference"""
        print("\n" + "=" * 60)
        print("INFERENCE RESULTS")
        print("=" * 60)

        if "best_threshold" in metrics:
            print(f"\n🎯 Best Threshold Search:")
            print(f"  Best Threshold: {metrics['best_threshold']:.6f}")
            print(f"  Best F1-Score: {metrics['best_f1']:.4f}")
            print(f"  Best Precision: {metrics['best_precision']:.4f}")
            print(f"  Best Recall: {metrics['best_recall']:.4f}")
            print(f"  Best Accuracy: {metrics['best_accuracy']:.4f}")
        else:
            print(f"\n🎯 Results (threshold={metrics['threshold']:.6f}):")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

        if "total_segments" in metrics and metrics["total_segments"] > 0:
            print(f"\n📈 Anomaly Segments:")
            print(f"  Total segments: {metrics['total_segments']}")
            print(f"  Detected segments: {metrics['detected_segments']}")
            print(f"  Detection rate: {metrics['segment_detection_rate']:.2%}")

        if "TP" in metrics:
            print(f"\n🔢 Confusion Matrix:")
            print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}")
            print(f"  FN: {metrics['FN']}, TN: {metrics['TN']}")

        print("=" * 60)


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Phase 2 Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="phase2/checkpoints/phase2_ucr_135_best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (optional, will use from checkpoint if not provided)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset (optional, will use from checkpoint if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference (optional, will use from checkpoint if not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu), auto-detect if not set",
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Phase 2: Inference Mode")
    print("=" * 60)

    # Load model
    inferencer = Phase2Inference(args.checkpoint, device=args.device)

    # Get config from checkpoint
    config = inferencer.config

    # Override dataset if provided
    dataset_name = args.dataset if args.dataset else config["dataset_name"]
    subset = args.subset if args.subset else config["subset"]
    batch_size = args.batch_size if args.batch_size else config["batch_size"]

    print(f"\n[1/3] Loading test data: {dataset_name}_{subset}")
    print(f"  Batch size: {batch_size}")

    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_base = os.path.join(project_root, "data", "datasets")

    dataloader_func = {
        "ucr": ucr_sub_ds_processing,
        "smd": smd_sub_ds_processing,
        "smap_msl": smap_msl_sub_ds_processing,
        "psm": psm_sub_ds_processing,
        "pd": pd_sub_ds_processing,
        "ecg": ecg_sub_ds_processing,
        "gesture": gesture_sub_ds_processing,
    }

    loader_func = dataloader_func[dataset_name]

    _, _, test_windows, test_labels, labels = prepare_phase2_data(
        dataset_name=dataset_name,
        subset=subset,
        loader_func=loader_func,
        window_size=config["window_size"],
        stride=config["stride"],
        data_path_base=data_path_base,
    )

    print(f"  ✓ Test set: {len(test_windows)} windows")
    print(f"  ✓ Time series length: {len(labels)}")
    print(f"  ✓ Anomaly rate: {labels.sum() / len(labels) * 100:.2f}%")

    # Create test dataloader only
    print("\n[2/3] Creating test dataloader...")
    test_dataset = Phase2Dataset(test_windows, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"  ✓ Test batches: {len(test_loader)}")

    # Run inference
    print("\n[3/3] Running inference...")
    metrics, anomaly_scores, labels = inferencer.predict(
        test_loader,
        stride=config["stride"],
        labels=labels,
    )

    # Print results
    inferencer.print_results(metrics)

    # Save results (optional)
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f"inference_{dataset_name}_{subset}_results.npz"
    )

    np.savez(output_file, anomaly_scores=anomaly_scores, labels=labels, metrics=metrics)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
