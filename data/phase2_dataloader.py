"""
Phase 2 Dataloader - For Supervised Anomaly Detection Training
Load specific dataset with train/test split and labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple
from data.sliding_window import create_sliding_windows


class Phase2Dataset(Dataset):
    """
    Dataset for Phase 2: Returns windows with labels
    Used for test/inference where labels are needed
    """
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data: shape (n_samples, n_channels, seq_len)
            labels: shape (n_samples,) - binary labels (0: normal, 1: anomaly)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        
        assert len(self.data) == len(self.labels), \
            f"Data and labels length mismatch: {len(self.data)} vs {len(self.labels)}"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Phase2TrainDataset(Dataset):
    """
    Simple dataset for Phase 2 training: Only returns data (no labels needed)
    Training data is all normal, so labels are not used in reconstruction task
    """
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: shape (n_samples, n_channels, seq_len)
        """
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def prepare_phase2_data(
    dataset_name: str,
    subset: str,
    loader_func,
    window_size: int,
    stride: int,
    data_path_base: str = "data/datasets"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare data for Phase 2 training
    
    Args:
        dataset_name: Dataset name (e.g., 'ucr', 'smap_msl_', 'ecg', 'pd', 'gesture')
        subset: Subset name (e.g., '135', '136', 'chfdb_chf01_275.pkl', None)
        loader_func: Data loading function
        window_size: Sliding window size
        stride: Sliding window stride
        data_path_base: Base path to datasets
    
    Returns:
        train_windows: (n_train, n_channels, window_size)
        train_labels: (n_train,)
        test_windows: (n_test, n_channels, window_size)
        test_labels: (n_test,) - window-level labels
        labels: (n_time_steps,) - test labels per time-step
    """
    # Load data
    # Different datasets have different path structures
    if dataset_name == "pd" or dataset_name == "ecg" or dataset_name == "psm" or dataset_name == "smd":
        # Gesture uses train/ and test/ directly, not labeled/
        data_path = os.path.join(data_path_base, dataset_name)
    else:
        # Other datasets use labeled/ subdirectory
        data_path = os.path.join(data_path_base, dataset_name, "labeled")
    train_data, test_data, labels = loader_func(
        data_path=data_path,
        filename=subset,
        normalized=True
    )
    
    print(f"\nLoaded {dataset_name}_{subset}:")
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Test data shape: {test_data.shape}")
    print(f"  Test labels shape: {labels.shape}")
    
    # Create sliding windows for train (all labels are 0 - normal)
    train_windows = create_sliding_windows(
        data=train_data,
        window_size=window_size,
        stride=stride
    )
    train_labels = np.zeros(train_windows.shape[0])  # All normal
    
    # Create sliding windows for test
    test_windows = create_sliding_windows(
        data=test_data,
        window_size=window_size,
        stride=stride
    )
    
    # Create labels for test windows
    # A window is anomalous if ANY point in it is anomalous
    test_window_labels = []
    for i in range(test_windows.shape[0]):
        start_idx = i * stride
        end_idx = start_idx + window_size
        window_labels = labels[start_idx:end_idx]
        # Label as anomaly if any point in window is anomaly
        is_anomaly = 1 if np.any(window_labels > 0) else 0
        test_window_labels.append(is_anomaly)
    
    test_window_labels = np.array(test_window_labels)
    
    print(f"\nWindows created:")
    print(f"  Train windows: {train_windows.shape}")
    print(f"  Train labels: {train_labels.shape} (normal: {np.sum(train_labels==0)})")
    print(f"  Test windows: {test_windows.shape}")
    print(f"  Test window labels: {test_window_labels.shape} (normal: {np.sum(test_window_labels==0)}, anomaly: {np.sum(test_window_labels==1)})")
    print(f"  Test labels: {labels.shape} (normal: {np.sum(labels==0)}, anomaly: {np.sum(labels==1)})")
    
    return train_windows, train_labels, test_windows, test_window_labels, labels


def create_phase2_dataloaders(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for Phase 2
    
    Args:
        train_windows: Training windows
        train_labels: Training labels (not used in training, but kept for compatibility)
        test_windows: Test windows
        test_labels: Test labels
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        train_loader, test_loader
    """
    import torch
    use_cuda = torch.cuda.is_available()
    
    # Training doesn't need labels (all normal data)
    train_dataset = Phase2TrainDataset(train_windows)
    # Test needs labels for evaluation
    test_dataset = Phase2Dataset(test_windows, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    from data.dataloader import ucr_sub_ds_processing
    
    # Test with UCR 135
    train_windows, train_labels, test_windows, test_labels, labels = prepare_phase2_data(
        dataset_name='ucr',
        subset='135',
        loader_func=ucr_sub_ds_processing,
        window_size=16,
        stride=1,
        data_path_base='data/datasets'
    )
    
    train_loader, test_loader = create_phase2_dataloaders(
        train_windows, train_labels,
        test_windows, test_labels,
        batch_size=32
    )
    
    # Test iteration - train loader only returns data (no labels)
    for batch_data in train_loader:
        print(f"\nTrain batch shape: {batch_data.shape}")
        break
    
    # Test iteration - test loader returns data and labels
    for batch_data, batch_labels in test_loader:
        print(f"\nTest batch shape: {batch_data.shape}")
        print(f"Test labels shape: {batch_labels.shape}")
        break

