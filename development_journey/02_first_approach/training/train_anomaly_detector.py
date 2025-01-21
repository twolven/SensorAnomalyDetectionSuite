"""
Training script for power system anomaly detection using TensorFlow.
Uses an autoencoder architecture to detect anomalies in power system waveforms.
"""

import os
import h5py
import json
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
from pathlib import Path

def create_autoencoder(sequence_length, n_features):
    """
    Create autoencoder model optimized for 6000Hz sampling rate power system data.
    """
    # Encoder - Deeper architecture with better downsampling
    encoder_inputs = layers.Input(shape=(sequence_length, n_features))
    
    # First encoding block - capture high-frequency components
    x = layers.Conv1D(128, 3, activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Second encoding block
    x = layers.Conv1D(64, 3, activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Third encoding block
    x = layers.Conv1D(32, 3, activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    # Decoder - Symmetric architecture
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoded)
    x = layers.Dropout(0.2)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    
    decoder_outputs = layers.Conv1D(n_features, 3, activation='tanh', padding='same')(x)

    # Create and compile model
    autoencoder = models.Model(encoder_inputs, decoder_outputs)
    
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=500,
        decay_rate=0.95,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    autoencoder.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder

def load_data(data_path):
    """
    Load training and validation data from HDF5 file.
    """
    with h5py.File(data_path, 'r') as f:
        x_train = f['train/data'][:]
        x_val = f['val/data'][:]
        
        print(f"Original shapes - Train: {x_train.shape}, Val: {x_val.shape}")
        
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)
        if len(x_val.shape) == 2:
            x_val = np.expand_dims(x_val, axis=-1)
            
        print(f"Reshaped - Train: {x_train.shape}, Val: {x_val.shape}")
        
        scaler = np.max(np.abs(x_train))
        x_train = x_train.astype('float32') / scaler
        x_val = x_val.astype('float32') / scaler
        print(f"Normalization scale factor: {scaler}")
        
        print(f"Training data stats - Min: {np.min(x_train)}, Max: {np.max(x_train)}, Mean: {np.mean(x_train)}, Std: {np.std(x_train)}")
        print(f"Validation data stats - Min: {np.min(x_val)}, Max: {np.max(x_val)}, Mean: {np.mean(x_val)}, Std: {np.std(x_val)}")
    
    return x_train, x_val

def calculate_threshold(model, normal_data, percentile=99):
    """
    Calculate anomaly threshold based on reconstruction error.
    """
    reconstructed = model.predict(normal_data)
    mse = np.mean(np.square(normal_data - reconstructed), axis=(1,2))
    
    print(f"MSE stats - Min: {np.min(mse)}, Max: {np.max(mse)}, Mean: {np.mean(mse)}, Std: {np.std(mse)}")
    
    threshold = np.percentile(mse, percentile)
    
    anomaly_rate = (mse > threshold).mean() * 100
    print(f"Training set anomaly rate with threshold {threshold:.4f}: {anomaly_rate:.1f}%")
    
    return threshold

def main():
    # Setup paths
    base_dir = Path("ml/training")
    data_path = base_dir / "data/tensorflow/anomaly_data.h5"
    model_path = base_dir / "models/tensorflow"
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    x_train, x_val = load_data(data_path)
    sequence_length = x_train.shape[1]
    n_features = x_train.shape[2]
    
    # Create and train model
    print("Creating model...")
    model = create_autoencoder(sequence_length, n_features)
    print(model.summary())
    
    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        callbacks.ModelCheckpoint(
            filepath=str(model_path / 'anomaly_detector.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        x_train, x_train,
        epochs=150,
        batch_size=32,
        validation_data=(x_val, x_val),
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True
    )
    
    # Calculate and save threshold
    print("Calculating anomaly threshold...")
    threshold = calculate_threshold(model, x_train)
    np.save(model_path / 'anomaly_threshold.npy', threshold)
    model.save(model_path / 'anomaly_detector.h5')
    
    # Save training history with NumPy type conversion
    history_dict = {}
    for key, value in history.history.items():
        history_dict[key] = [float(val) for val in value]  # Convert numpy values to Python floats
    
    with open(model_path / 'training_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    print("Training complete!")

if __name__ == "__main__":
    main()