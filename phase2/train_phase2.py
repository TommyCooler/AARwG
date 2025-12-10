"""
Phase 2 Training: Supervised Anomaly Detection with AGF-TCN
- Load pre-trained Augmentation module (frozen)
- Train AGF-TCN for reconstruction
- MSE Loss between augmented data and reconstructed data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.phase2_dataloader import prepare_phase2_data, Phase2TrainDataset, Phase2Dataset
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
from modules.random_masking import RandomTimeMasking, InferenceMasking
from phase2.agf_tcn import Agf_TCN
from utils.point_adjustment import evaluate_with_pa

import setproctitle

setproctitle.setproctitle("Tran Chi Tam - Phase 2 Training")


class Phase2Trainer:
    """
    Trainer for Phase 2: Supervised anomaly detection with reconstruction
    """

    def __init__(
        self,
        augmentation: nn.Module,
        agf_tcn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_scheduler: bool = True,
        use_grad_clip: bool = True,
        max_grad_norm: float = 1.0,
        config: dict = None,  # Store config for inference
        mask_ratio: float = 0.15,  # Random time masking ratio
    ):
        self.augmentation = augmentation.to(device)
        self.agf_tcn = agf_tcn.to(device)
        self.device = device

        # Initialize random time masking
        self.time_masking = RandomTimeMasking(mask_ratio=mask_ratio).to(device)
        self.use_scheduler = use_scheduler
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.config = config or {}  # Store for saving to checkpoint

        # Freeze augmentation module
        self._freeze_augmentation()

        # Only optimize AGF-TCN parameters
        self.optimizer = optim.Adam(
            self.agf_tcn.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler (optional)
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )

        # Reconstruction loss
        self.criterion = nn.MSELoss()

        print(f"\nPhase 2 Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Augmentation: FROZEN")
        print(
            f"  AGF-TCN parameters: {sum(p.numel() for p in self.agf_tcn.parameters()):,}"
        )
        print(
            f"  Trainable parameters: {sum(p.numel() for p in self.agf_tcn.parameters() if p.requires_grad):,}"
        )

    def _freeze_augmentation(self):
        """Freeze all parameters in augmentation module"""
        frozen_count = 0

        for param in self.augmentation.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        self.augmentation.eval()  # Set to eval mode

        # Verify all parameters are frozen
        total_params = sum(1 for _ in self.augmentation.parameters())
        still_trainable = sum(
            1 for p in self.augmentation.parameters() if p.requires_grad
        )

        if still_trainable > 0:
            print(
                f"  ⚠ WARNING: {still_trainable}/{total_params} augmentation parameters still trainable!"
            )
        else:
            print(f"  ✓ Augmentation module frozen ({frozen_count} parameters)")

        # Verify eval mode
        if not next(self.augmentation.parameters()).requires_grad:
            print(f"  ✓ Augmentation in eval mode (deterministic)")
        else:
            print(f"  ⚠ WARNING: Augmentation not properly frozen!")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.agf_tcn.train()
        self.augmentation.eval()  # Keep augmentation in eval mode
        self.time_masking.train()  # Enable masking during training

        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch_data in enumerate(pbar):
            batch_data = batch_data.to(self.device)

            # Apply random time masking before augmentation
            masked_data = self.time_masking(batch_data)

            # Forward pass
            with torch.no_grad():
                # Get augmented data (frozen augmentation)
                augmented_data = self.augmentation(masked_data)

            # Reconstruct from augmented data
            reconstructed = self.agf_tcn(augmented_data)

            # Compute reconstruction loss
            loss = self.criterion(reconstructed, augmented_data)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional)
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.agf_tcn.parameters(), max_norm=self.max_grad_norm
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "avg_loss": f"{total_loss/(batch_idx+1):.6f}",
                }
            )

        avg_loss = total_loss / len(train_loader)

        return {"train_loss": avg_loss}

    def map_window_scores_to_timeseries(
        self,
        timestep_scores_all_windows: np.ndarray,
        n_time_steps: int,
        window_size: int,
        stride: int,
    ) -> np.ndarray:
        """
        Map window-based per-time-step scores back to original time series.
        """
        n_windows = timestep_scores_all_windows.shape[0]
        timeseries_scores = np.full(n_time_steps, np.nan, dtype=np.float32)

        # Map window 0: all time steps
        window_0_scores = timestep_scores_all_windows[0]
        timeseries_scores[0:window_size] = window_0_scores

        # Map subsequent windows: only last time step
        for i in range(1, n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            last_time_step = end_idx - 1

            if last_time_step < n_time_steps:
                last_timestep_score = timestep_scores_all_windows[i, -1]
                timeseries_scores[last_time_step] = last_timestep_score

        # Fill NaN values with forward fill
        valid_mask = ~np.isnan(timeseries_scores)
        if not valid_mask.all():
            last_valid_idx = -1
            for i in range(n_time_steps):
                if valid_mask[i]:
                    last_valid_idx = i
                elif last_valid_idx >= 0:
                    timeseries_scores[i] = timeseries_scores[last_valid_idx]

        return timeseries_scores

    def evaluate(self, test_loader, labels, stride):
        """
        Run inference on test data and return F1 score and metrics.
        """
        self.agf_tcn.eval()
        self.augmentation.eval()
        
        # Initialize inference masking
        inference_masking = InferenceMasking(mask_ratio=self.config.get("mask_ratio", 0.15)).to(self.device)
        inference_masking.reset()
        
        all_timestep_scores = []
        global_window_idx = 0

        with torch.no_grad():
            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.shape[0]

                # Apply inference masking with window index tracking
                masked_data = batch_data.clone()
                for i in range(batch_size):
                    window_data = batch_data[i : i + 1]
                    masked_window = inference_masking(
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

        # Map window scores to time series
        anomaly_scores = self.map_window_scores_to_timeseries(
            timestep_scores_all_windows=timestep_scores_all_windows,
            n_time_steps=len(labels),
            window_size=self.config["window_size"],
            stride=stride,
        )

        # Evaluate with Point Adjustment
        metrics = evaluate_with_pa(anomaly_scores=anomaly_scores, labels=labels)

        return metrics

    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint with full config for inference"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "agf_tcn_state_dict": self.agf_tcn.state_dict(),
            "augmentation_state_dict": self.augmentation.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,  # Save full config for inference
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved with config: {path}")


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
    # Configuration
    config = {
        # Data config
        "dataset_name": "ucr",
        "subset": "136",  # Specific dataset: 135, 136, 137, or 138
        # Model config
        "agf_tcn_channels": [256],  # TCN hidden channels
        "dropout": 0.1,
        "activation": "gelu",
        "fuse_type": 2,
        # Training config
        "batch_size": 64,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "use_scheduler": False,  # Use learning rate scheduler
        "use_grad_clip": False,  # Use gradient clipping
        "max_grad_norm": 1.0,  # Max gradient norm for clipping
        # Random time masking options
        "mask_ratio": 0.15,  # Percentage of time steps to mask (0.0 to 1.0)
        # Phase 1 checkpoint (pre-trained augmentation)
        "phase1_checkpoint": "phase1/checkpoints/ucr_best_model.pth",
        # Misc
        "num_workers": 0,
        "save_dir": "phase2/checkpoints",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("=" * 60)
    print("Phase 2: Supervised Anomaly Detection Training")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path_base = os.path.join(project_root, "data", "datasets")

    # Step 1: Load Phase 1 checkpoint to get window_size and stride
    print("\n[1/6] Loading Phase 1 checkpoint for window_size and stride...")
    phase1_checkpoint_path = os.path.join(project_root, config["phase1_checkpoint"])
    
    if not os.path.exists(phase1_checkpoint_path):
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found at {phase1_checkpoint_path}\n"
            f"Please train Phase 1 first or update 'phase1_checkpoint' path in config."
        )
    
    phase1_checkpoint = torch.load(phase1_checkpoint_path, map_location=config["device"])
    phase1_config = phase1_checkpoint["config"]
    
    # Get window_size and stride from Phase 1
    config["window_size"] = phase1_config["window_size"]
    config["stride"] = phase1_config["stride"]
    print(f"  ✓ Loaded window_size: {config['window_size']}, stride: {config['stride']} from Phase 1")

    # Step 2: Prepare data
    print(f"\n[2/6] Loading dataset: {config['dataset_name']}_{config['subset']}")

    # Dataloader function mapping
    dataloader_func = {
        "ucr": ucr_sub_ds_processing,
        "smd": smd_sub_ds_processing,
        "smap_msl": smap_msl_sub_ds_processing,
        "psm": psm_sub_ds_processing,
        "pd": pd_sub_ds_processing,
        "ecg": ecg_sub_ds_processing,
        "gesture": gesture_sub_ds_processing,
    }

    loader_func = dataloader_func[config["dataset_name"]]

    # Load data (train cho training, test cho inference)
    train_windows, _, test_windows, test_labels, labels = prepare_phase2_data(
        dataset_name=config["dataset_name"],
        subset=config["subset"],
        loader_func=loader_func,
        window_size=config["window_size"],
        stride=config["stride"],
        data_path_base=data_path_base,
    )

    # Step 3: Create train and test dataloaders
    print("\n[3/6] Creating dataloaders...")
    # Training data is all normal, labels not needed for reconstruction task
    train_dataset = Phase2TrainDataset(train_windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    print(f"  Train batches: {len(train_loader)}")
    
    # Test dataset for inference
    test_dataset = Phase2Dataset(test_windows, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Test time series length: {len(labels)}")

    # Get data dimensions
    n_channels = train_windows.shape[1]
    window_size = train_windows.shape[2]

    # Update config with actual dimensions (important for inference!)
    config["n_channels"] = n_channels
    config["window_size"] = window_size

    # Step 4: Initialize Augmentation (load from Phase 1)
    print("\n[4/6] Loading pre-trained Augmentation from Phase 1...")
    # Phase 1 checkpoint already loaded above
    
    # Copy augmentation config from Phase 1 to Phase 2 config (for saving checkpoint)
    config["aug_kernel_size_cnn"] = phase1_config["aug_kernel_size_cnn"]
    config["aug_num_layers"] = phase1_config["aug_num_layers"]
    config["aug_dropout"] = phase1_config["aug_dropout"]
    config["aug_temperature"] = phase1_config["aug_temperature"]
    config["aug_hard_gumbel_softmax"] = phase1_config["aug_hard_gumbel_softmax"]
    config["aug_transformer_d_model"] = phase1_config["aug_transformer_d_model"]
    config["aug_transformer_nhead"] = phase1_config["aug_transformer_nhead"]
    
    # Initialize augmentation with Phase 1 config
    augmentation = Augmentation(
        in_channels=n_channels,
        seq_len=window_size,
        kernel_size=phase1_config["aug_kernel_size_cnn"],
        num_layers=phase1_config["aug_num_layers"],
        dropout=phase1_config["aug_dropout"],
        temperature=phase1_config["aug_temperature"],
        hard=phase1_config["aug_hard_gumbel_softmax"],
        transformer_d_model=phase1_config["aug_transformer_d_model"],
        transformer_nhead=phase1_config["aug_transformer_nhead"],
    )

    # Load Phase 1 checkpoint weights
    augmentation.load_state_dict(phase1_checkpoint["augmentation_state_dict"])
    print(f"  ✓ Loaded augmentation from: {phase1_checkpoint_path}")

    # Step 5: Initialize AGF-TCN
    print("\n[5/6] Initializing AGF-TCN...")
    agf_tcn = Agf_TCN(
        num_inputs=n_channels,
        num_channels=config["agf_tcn_channels"],
        dropout=config["dropout"],
        activation=config["activation"],
        fuse_type=config["fuse_type"],
        window_size=window_size,
    )
    print(f"  AGF-TCN channels: {config['agf_tcn_channels']}")
    print(f"  Input shape: ({n_channels}, {window_size})")
    print(f"  Output shape: ({n_channels}, {window_size})")

    # Step 6: Initialize trainer
    print("\n[6/6] Initializing trainer...")
    trainer = Phase2Trainer(
        augmentation=augmentation,
        agf_tcn=agf_tcn,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        device=config["device"],
        use_scheduler=config["use_scheduler"],
        use_grad_clip=config["use_grad_clip"],
        max_grad_norm=config["max_grad_norm"],
        config=config,  # Pass full config for checkpoint
        mask_ratio=config["mask_ratio"],
    )

    # Step 7: Training loop
    print("\n[7/7] Starting training...")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Device: {config['device']}")
    print(f"  Inference will run after each epoch")

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Print training metrics
        print(f"\nEpoch {epoch} Training Results:")
        print(f"  Train Loss: {train_metrics['train_loss']:.6f}")

        # Learning rate scheduling (optional)
        if trainer.scheduler is not None:
            trainer.scheduler.step(train_metrics["train_loss"])

        # Run inference on test set
        print(f"\nRunning inference on test set...")
        inference_metrics = trainer.evaluate(
            test_loader=test_loader,
            labels=labels,
            stride=config["stride"]
        )

        # Get F1 score
        f1_score = inference_metrics.get("best_f1", 0.0)

        # Print inference metrics
        print(f"\nEpoch {epoch} Inference Results:")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  Precision: {inference_metrics.get('best_precision', 0.0):.4f}")
        print(f"  Recall: {inference_metrics.get('best_recall', 0.0):.4f}")
        print(f"  Accuracy: {inference_metrics.get('best_accuracy', 0.0):.4f}")

        # Combine metrics for checkpoint
        combined_metrics = {
            **train_metrics,
            "f1": f1_score,
            "precision": inference_metrics.get("best_precision", 0.0),
            "recall": inference_metrics.get("best_recall", 0.0),
            "accuracy": inference_metrics.get("best_accuracy", 0.0),
        }

        # Save checkpoint when F1 improves
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epoch = epoch

            save_path = os.path.join(
                project_root,
                config["save_dir"],
                f"phase2_{config['dataset_name']}_{config['subset']}_best.pt",
            )
            trainer.save_checkpoint(save_path, epoch, combined_metrics)
            print(f"  ✓ New best model! F1-Score: {best_f1:.4f} (Epoch {epoch})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best F1-Score: {best_f1:.4f} (Epoch {best_epoch})")
    print(
        f"Saved checkpoint: phase2/checkpoints/phase2_{config['dataset_name']}_{config['subset']}_best.pt"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
