"""
Training data generation script for power system anomaly detection.
Generates datasets for TensorFlow, PyTorch, and YOLO models with different fault scenarios.
"""

import os
import numpy as np
from pathlib import Path
import h5py
from augmentation.keras_augmentor import PowerSystemAugmentor, DatasetGenerator

def create_output_dirs():
    """Create directory structure for output data"""
    base_dir = Path("ml/training/data")
    dirs = [
        base_dir / "tensorflow",
        base_dir / "pytorch",
        base_dir / "yolo",
        base_dir / "validation"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    return base_dir

def generate_tensorflow_data(augmentor, base_dir, n_samples=10000):
    """
    Generate time-series data for TensorFlow anomaly detection.
    Creates sequences of normal operation with rare anomalies.
    """
    print("Generating TensorFlow training data...")
    
    # Generate mostly normal data with fewer faults for anomaly detection
    generator = DatasetGenerator(augmentor, sequence_length=128, stride=16)
    train_ds, val_ds = generator.generate_training_data(
        n_samples=n_samples, 
        validation_split=0.2
    )
    
    # Save datasets
    output_path = base_dir / "tensorflow"
    with h5py.File(output_path / "anomaly_data.h5", 'w') as f:
        # Convert datasets to numpy arrays
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        
        # Collect training data
        for x, y in train_ds:
            x_train.append(x.numpy())
            y_train.append(y.numpy())
            
        # Collect validation data
        for x, y in val_ds:
            x_val.append(x.numpy())
            y_val.append(y.numpy())
        
        # Concatenate and save
        if x_train:
            f.create_dataset("train/data", data=np.concatenate(x_train))
            f.create_dataset("train/labels", data=np.concatenate(y_train))
        if x_val:
            f.create_dataset("val/data", data=np.concatenate(x_val))
            f.create_dataset("val/labels", data=np.concatenate(y_val))

def generate_pytorch_data(augmentor, base_dir, n_samples=10000):
    """
    Generate fault classification data for PyTorch.
    Creates balanced dataset of different fault types.
    """
    print("Generating PyTorch training data...")
    
    # Generate balanced fault data
    x_data = []
    y_data = []
    fault_types = ['normal', 'sag', 'swell', 'interruption', 'harmonic']
    samples_per_class = n_samples // len(fault_types)
    
    for fault_idx, _ in enumerate(fault_types):
        waves, _ = augmentor.generate_batch(
            batch_size=samples_per_class,
            include_faults=True
        )
        x_data.append(waves)
        y_data.append(np.full(samples_per_class, fault_idx))
    
    x_data = np.concatenate(x_data)
    y_data = np.concatenate(y_data)
    
    # Random shuffle
    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train/val
    split_idx = int(0.8 * len(x_data))
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    
    # Save datasets
    output_path = base_dir / "pytorch"
    with h5py.File(output_path / "fault_classification.h5", 'w') as f:
        f.create_dataset("train/data", data=x_train)
        f.create_dataset("train/labels", data=y_train)
        f.create_dataset("val/data", data=x_val)
        f.create_dataset("val/labels", data=y_val)
        # Save fault type mapping
        dt = h5py.special_dtype(vlen=str)
        fault_types_ds = f.create_dataset('fault_types', (len(fault_types),), dtype=dt)
        fault_types_ds[:] = fault_types

def generate_yolo_data(augmentor, base_dir, n_samples=5000):
    """
    Generate waveform pattern data for YOLO analysis.
    Creates sequences with multiple fault patterns for detection.
    """
    print("Generating YOLO training data...")
    
    # Generate longer sequences with multiple faults
    window_size = 1024  # Longer window for pattern detection
    augmentor_long = PowerSystemAugmentor(window_size=window_size)
    
    x_data = []
    annotations = []
    
    for _ in range(n_samples):
        # Generate base waveform
        wave = augmentor_long.generate_base_waveform()
        
        # Always add 1-2 faults for more consistent training
        n_faults = np.random.randint(1, 3)
        fault_annotations = []
        
        # Track used positions to ensure fault separation
        used_positions = []
        
        for _ in range(n_faults):
            fault_type = np.random.choice(['sag', 'swell', 'interruption', 'harmonic'])
            # Stronger, more detectable faults
            magnitude = np.random.uniform(0.4, 0.8)
            duration = np.random.randint(50, 120)  # Longer duration for better detection
            
            # Ensure minimum separation between faults
            min_separation = window_size // 4
            valid_start = False
            max_attempts = 50
            attempts = 0
            
            while not valid_start and attempts < max_attempts:
                start_idx = np.random.randint(0, window_size - duration)
                valid_start = True
                for used_pos, used_dur in used_positions:
                    if abs(start_idx - used_pos) < min_separation:
                        valid_start = False
                        break
                attempts += 1
            
            if valid_start:
                used_positions.append((start_idx, duration))
                fault_annotations.append({
                    'type': fault_type,
                    'start': start_idx / window_size,  # Normalize to 0-1
                    'width': duration / window_size,   # Normalize to 0-1
                    'magnitude': magnitude
                })
                
                # Inject the fault
                wave = augmentor_long.inject_fault(
                    wave, fault_type, magnitude, duration_samples=duration
                )
        
        x_data.append(wave)
        annotations.append(fault_annotations)
    
    # Split into train/val
    split_idx = int(0.8 * len(x_data))
    x_train = np.array(x_data[:split_idx])
    x_val = np.array(x_data[split_idx:])
    ann_train = annotations[:split_idx]
    ann_val = annotations[split_idx:]
    
    # Save datasets
    output_path = base_dir / "yolo"
    with h5py.File(output_path / "pattern_detection.h5", 'w') as f:
        f.create_dataset("train/data", data=x_train)
        f.create_dataset("val/data", data=x_val)
        
        # Save annotations as JSON-compatible strings
        import json
        dt = h5py.special_dtype(vlen=str)
        ann_train_ds = f.create_dataset('train/annotations', (len(ann_train),), dtype=dt)
        ann_val_ds = f.create_dataset('val/annotations', (len(ann_val),), dtype=dt)
        ann_train_ds[:] = [json.dumps(ann) for ann in ann_train]
        ann_val_ds[:] = [json.dumps(ann) for ann in ann_val]

def main():
    """Main data generation function"""
    print("Starting training data generation...")
    
    # Create output directories
    base_dir = create_output_dirs()
    
    # Initialize augmentor
    augmentor = PowerSystemAugmentor()
    
    # Generate data for each model
    generate_tensorflow_data(augmentor, base_dir)
    generate_pytorch_data(augmentor, base_dir)
    generate_yolo_data(augmentor, base_dir)
    
    print("Training data generation complete!")

if __name__ == "__main__":
    main()