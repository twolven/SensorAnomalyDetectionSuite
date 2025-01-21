"""
Validation script for the trained anomaly detector.
Visualizes reconstructions and error distributions to verify model performance.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data_and_model():
    """Load test data and trained model"""
    base_dir = Path("ml/training")
    
    # Load test data
    with h5py.File(base_dir / "data/tensorflow/anomaly_data.h5", 'r') as f:
        x_val = f['val/data'][:]
        x_train = f['train/data'][:]
        
        # Reshape if needed
        if len(x_val.shape) == 2:
            x_val = np.expand_dims(x_val, axis=-1)
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)
            
        # Normalize using training data max
        scaler = np.max(np.abs(x_train))
        x_val = x_val.astype('float32') / scaler
        print(f"Validation data stats - Min: {np.min(x_val)}, Max: {np.max(x_val)}, "
              f"Mean: {np.mean(x_val)}, Std: {np.std(x_val)}")

    # Define custom objects
    class CustomMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
        def __init__(self, name='mean_absolute_error', dtype=None):
            super().__init__(name=name, dtype=dtype)
        
        @property
        def name(self):
            return self._name if hasattr(self, '_name') else 'mean_absolute_error'

    custom_objects = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'mse': tf.keras.losses.MeanSquaredError(),
        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
        'mae': CustomMeanAbsoluteError,
        'mean_absolute_error': CustomMeanAbsoluteError,
        'MeanAbsoluteError': CustomMeanAbsoluteError
    }
    
    try:
        # Try loading with compile=False first
        model = keras.models.load_model(
            base_dir / "models/tensorflow/anomaly_detector.h5",
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    try:
        threshold = np.load(base_dir / "models/tensorflow/anomaly_threshold.npy")
    except Exception as e:
        print(f"Error loading threshold: {str(e)}")
        raise
    
    return x_val, model, threshold

def plot_reconstructions(original, reconstructed, indices, threshold):
    """Plot original vs reconstructed waveforms"""
    fig, axes = plt.subplots(len(indices), 1, figsize=(15, 4*len(indices)))
    if len(indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Calculate reconstruction error
        error = np.mean(np.square(original[idx] - reconstructed[idx]))
        is_anomaly = error > threshold
        
        # Plot original and reconstruction
        axes[i].plot(original[idx], label='Original', alpha=0.7)
        axes[i].plot(reconstructed[idx], label='Reconstructed', alpha=0.7)
        axes[i].set_title(f'Sample {idx} - Error: {error:.4f} {"(Anomaly)" if is_anomaly else "(Normal)"}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('reconstruction_samples.png')
    plt.close()

def plot_error_distribution(original, reconstructed, threshold):
    """Plot distribution of reconstruction errors"""
    # Calculate reconstruction errors
    errors = np.mean(np.square(original - reconstructed), axis=(1,2))
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.close()
    
    # Print statistics
    anomaly_rate = (errors > threshold).mean() * 100
    print(f"\nError Statistics:")
    print(f"Mean Error: {errors.mean():.4f}")
    print(f"Std Error: {errors.std():.4f}")
    print(f"Anomaly Rate: {anomaly_rate:.1f}%")
    print(f"Threshold: {threshold:.4f}")

def main():
    print("Loading data and model...")
    x_val, model, threshold = load_data_and_model()
    
    print("Generating reconstructions...")
    reconstructed = model.predict(x_val)
    
    # Plot a few samples (normal and potentially anomalous)
    print("Plotting sample reconstructions...")
    errors = np.mean(np.square(x_val - reconstructed), axis=(1,2))
    normal_idx = np.argmin(errors)  # Most normal sample
    anomaly_idx = np.argmax(errors)  # Most anomalous sample
    random_idx = np.random.randint(0, len(x_val))  # Random sample
    
    plot_reconstructions(x_val, reconstructed, [normal_idx, anomaly_idx, random_idx], threshold)
    
    print("Plotting error distribution...")
    plot_error_distribution(x_val, reconstructed, threshold)

if __name__ == "__main__":
    main()