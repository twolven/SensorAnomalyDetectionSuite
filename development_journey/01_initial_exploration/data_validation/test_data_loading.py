"""
Test script to verify data loading and basic PyTorch setup
"""

import os
import h5py
import numpy as np
from pathlib import Path

def test_data_loading():
    try:
        data_path = Path("ml/training/data/pytorch/fault_classification.h5")
        print(f"Checking if file exists: {data_path.exists()}")
        
        with h5py.File(data_path, 'r') as f:
            print("\nFile structure:")
            def print_structure(name, obj):
                print(f"- {name}: {type(obj)}")
            f.visititems(print_structure)
            
            if 'train/data' in f:
                x_train = f['train/data'][:]
                y_train = f['train/labels'][:]
                print(f"\nTraining data shape: {x_train.shape}")
                print(f"Training labels shape: {y_train.shape}")
                print(f"Data type: {x_train.dtype}")
                print(f"Labels type: {y_train.dtype}")
                print(f"Data range: [{x_train.min()}, {x_train.max()}]")
                print(f"Unique labels: {np.unique(y_train)}")
            else:
                print("Could not find train/data in file")
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting data loading test...")
    test_data_loading()