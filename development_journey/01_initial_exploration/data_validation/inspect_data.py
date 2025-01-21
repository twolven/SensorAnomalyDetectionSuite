"""
Script to inspect the generated training data files.
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_tensorflow_data():
    """Inspect the anomaly detection dataset"""
    print("\n=== TensorFlow Anomaly Detection Data ===")
    with h5py.File('data/tensorflow/anomaly_data.h5', 'r') as f:
        print("\nDataset structure:")
        print_structure(f)
        
        # Show sample shapes and stats
        if 'train/data' in f:
            train_data = f['train/data'][:]
            train_labels = f['train/labels'][:]
            print(f"\nTraining data shape: {train_data.shape}")
            print(f"Training labels shape: {train_labels.shape}")
            print(f"Label distribution: {np.unique(train_labels, return_counts=True)}")
            
            # Plot a sample
            plt.figure(figsize=(10, 4))
            plt.plot(train_data[0])
            plt.title("Sample Anomaly Detection Sequence")
            plt.savefig("sample_anomaly.png")
            plt.close()

def inspect_pytorch_data():
    """Inspect the fault classification dataset"""
    print("\n=== PyTorch Fault Classification Data ===")
    with h5py.File('data/pytorch/fault_classification.h5', 'r') as f:
        print("\nDataset structure:")
        print_structure(f)
        
        # Show sample shapes and stats
        if 'train/data' in f:
            train_data = f['train/data'][:]
            train_labels = f['train/labels'][:]
            print(f"\nTraining data shape: {train_data.shape}")
            print(f"Training labels shape: {train_labels.shape}")
            print(f"Label distribution: {np.unique(train_labels, return_counts=True)}")
            
            # Plot samples for each class
            fault_types = [x.decode('utf-8') for x in f['fault_types'][:]]
            fig, axes = plt.subplots(len(fault_types), 1, figsize=(12, 3*len(fault_types)))
            for i, fault_type in enumerate(fault_types):
                idx = np.where(train_labels == i)[0][0]
                axes[i].plot(train_data[idx])
                axes[i].set_title(f"Sample {fault_type} waveform")
            plt.tight_layout()
            plt.savefig("sample_faults.png")
            plt.close()

def inspect_yolo_data():
    """Inspect the pattern detection dataset"""
    print("\n=== YOLO Pattern Detection Data ===")
    with h5py.File('data/yolo/pattern_detection.h5', 'r') as f:
        print("\nDataset structure:")
        print_structure(f)
        
        # Show sample shapes and annotations
        if 'train/data' in f:
            train_data = f['train/data'][:]
            annotations = f['train/annotations'][:]
            print(f"\nTraining data shape: {train_data.shape}")
            print(f"Number of sequences: {len(annotations)}")
            
            # Print sample annotation
            import json
            print("\nSample annotation:")
            print(json.loads(annotations[0]))
            
            # Plot a sample with annotations
            plt.figure(figsize=(12, 4))
            plt.plot(train_data[0])
            plt.title("Sample Pattern Detection Sequence with Annotations")
            plt.savefig("sample_pattern.png")
            plt.close()

def print_structure(hdf_file, level=0):
    """Recursively print HDF5 file structure"""
    for key in hdf_file.keys():
        print("  " * level + f"/{key}")
        if isinstance(hdf_file[key], h5py.Group):
            print_structure(hdf_file[key], level + 1)

def main():
    print("Inspecting generated datasets...")
    try:
        inspect_tensorflow_data()
    except Exception as e:
        print(f"Error inspecting TensorFlow data: {e}")
    
    try:
        inspect_pytorch_data()
    except Exception as e:
        print(f"Error inspecting PyTorch data: {e}")
    
    try:
        inspect_yolo_data()
    except Exception as e:
        print(f"Error inspecting YOLO data: {e}")

if __name__ == "__main__":
    main()