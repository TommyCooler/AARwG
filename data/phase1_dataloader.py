import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import sys

from .sliding_window import create_sliding_windows

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Phase1TrainDataset(Dataset):
    """
    Simple dataset for Phase 1 training: Only returns windows (no labels needed)
    Training data is all normal, used for contrastive learning
    """
    def __init__(self, all_windows: np.ndarray):
        self.windows = torch.from_numpy(all_windows).float()
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx]


def prepare_phase1_data(
    datasets_info: List[dict],
    window_size: int,
    stride: int,
    dataloader_func,
    data_path_base: str = "data/datasets"
) -> Tuple[np.ndarray, dict]:
    
    all_train_windows = []
    stats = {
        'datasets': [],
        'n_windows_per_dataset': [],
        'total_windows': 0
    }
    
    for idx, ds_info in enumerate(datasets_info):
        dataset_name = ds_info['name']
        subset = ds_info.get('subset', None)
        loader_name = ds_info['loader']
        
        # Load data using appropriate loader
        # Different datasets have different path structures
        if loader_name == "pd" or loader_name == "ecg" or loader_name == "psm" or loader_name == "smd":
            # Gesture uses train/ and test/ directly, not labeled/
            data_path = os.path.join(data_path_base, dataset_name)
        else:
            # Other datasets use labeled/ subdirectory
            data_path = os.path.join(data_path_base, dataset_name, "labeled")
        
        loader_func = dataloader_func[loader_name]
        
        # Load train data (không cần test labels trong phase 1)
        train_data, _, _ = loader_func(
            data_path=data_path,
            filename=subset,
            normalized=True
        )
        
        # Create sliding windows
        train_windows = create_sliding_windows(
            data=train_data,
            window_size=window_size,
            stride=stride
        )
        
        # Add to collection
        all_train_windows.append(train_windows)
        
        # Update stats
        stats['datasets'].append(f"{dataset_name}_{subset}" if subset else dataset_name)
        stats['n_windows_per_dataset'].append(train_windows.shape[0])
    
    # Concatenate all windows
    all_train_windows = np.concatenate(all_train_windows, axis=0)
    stats['total_windows'] = all_train_windows.shape[0]
    stats['shape'] = all_train_windows.shape
    
    # Shuffle windows
    shuffle_indices = np.random.permutation(stats['total_windows'])
    all_train_windows = all_train_windows[shuffle_indices]
    
    return all_train_windows, stats


def create_phase1_dataloader(
    all_train_windows: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    
    dataset = Phase1TrainDataset(all_train_windows)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader


# Example usage / Demo
if __name__ == "__main__":
    # Create fake data for demo
    window_size = 50
    stride = 10
    
    # Simulate 4 UCR subsets
    datasets_data = []
    for subset_id in [135, 136, 137, 138]:
        n_channels = 1
        n_windows = np.random.randint(40, 100)
        time_steps = window_size + (n_windows * stride)
        train_data = np.random.randn(n_channels, time_steps)
        datasets_data.append((subset_id, train_data))
    
    # Prepare windows from all datasets
    all_train_windows = []
    
    for subset_id, train_data in datasets_data:
        
        train_windows = create_sliding_windows(
            data=train_data,
            window_size=window_size,
            stride=stride
        )
        all_train_windows.append(train_windows)
    
    # Concatenate and shuffle
    all_train_windows = np.concatenate(all_train_windows, axis=0)
    shuffle_indices = np.random.permutation(len(all_train_windows))
    all_train_windows = all_train_windows[shuffle_indices]
    
    # Create DataLoader
    dataloader = create_phase1_dataloader(
        all_train_windows,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )
    
    print(f"Phase 1 DataLoader created:")
    print(f"  Total windows: {len(all_train_windows)}")
    print(f"  Batches: {len(dataloader)}")
    print(f"  Shape per batch: {next(iter(dataloader)).shape}")

