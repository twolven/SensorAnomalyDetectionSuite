"""
Power system data augmentation using modern Keras 3.
Provides utilities for generating synthetic power system data and augmenting with realistic faults.
"""

import numpy as np
from keras import layers
from keras.src.utils import timeseries_dataset_from_array
import scipy.signal as signal

class PowerSystemAugmentor:
    def __init__(self, sampling_rate=6000, window_size=1000):
        """
        Initialize the power system data augmentor with optimized parameters.
        
        Args:
            sampling_rate (int): Samples per second (default 6000 for better 60Hz resolution)
            window_size (int): Number of samples in each window (default 1000 for better resolution)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.noise_layer = layers.GaussianNoise(0.01)
        
    def generate_base_waveform(self, frequency=60.0, amplitude=1.0, phase=0.0):
        """
        Generate a base power system waveform with accurate frequency representation.
        
        Args:
            frequency (float): Frequency in Hz
            amplitude (float): Signal amplitude
            phase (float): Phase offset in radians
            
        Returns:
            numpy.ndarray: Generated waveform
        """
        duration = self.window_size / self.sampling_rate
        t = np.linspace(0, duration, self.window_size, endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def add_harmonics(self, waveform, harmonic_amplitudes):
        """
        Add harmonic components to the base waveform.
        
        Args:
            waveform (numpy.ndarray): Base waveform
            harmonic_amplitudes (dict): Dictionary of harmonic numbers and their relative amplitudes
            
        Returns:
            numpy.ndarray: Waveform with harmonics
        """
        duration = self.window_size / self.sampling_rate
        t = np.linspace(0, duration, self.window_size, endpoint=False)
        result = waveform.copy()
        
        for harmonic, amplitude in harmonic_amplitudes.items():
            result += amplitude * np.sin(2 * np.pi * 60 * harmonic * t)
            
        return result
    
    def add_noise(self, waveform, snr_db=40):
        """
        Add Gaussian noise to the waveform using Keras GaussianNoise layer.
        
        Args:
            waveform (numpy.ndarray): Input waveform
            snr_db (float): Signal-to-noise ratio in dB
            
        Returns:
            numpy.ndarray: Noisy waveform
        """
        signal_power = np.mean(waveform ** 2)
        noise_stddev = np.sqrt(signal_power / (10 ** (snr_db / 10)))
        self.noise_layer.stddev = noise_stddev
        
        # Convert to tensor, add batch and channel dimensions
        waveform_tensor = np.expand_dims(np.expand_dims(waveform, 0), -1)
        noisy_waveform = self.noise_layer(waveform_tensor, training=True)
        
        return np.squeeze(noisy_waveform)
    
    def inject_fault(self, waveform, fault_type, magnitude=0.5, duration_samples=None):
        """
        Inject a power system fault into the waveform with more distinctive characteristics.
        
        Args:
            waveform (numpy.ndarray): Input waveform
            fault_type (str): Type of fault ('sag', 'swell', 'interruption', 'harmonic')
            magnitude (float): Relative magnitude of the fault
            duration_samples (int): Duration of the fault in samples. If None, scales with window size.
            
        Returns:
            numpy.ndarray: Waveform with injected fault
        """
        result = waveform.copy()
        
        # Scale duration with window size if not specified
        if duration_samples is None:
            # Calculate duration to be between 2-4 cycles at 60Hz for more noticeable faults
            samples_per_cycle = self.sampling_rate / 60
            duration_samples = int(np.random.uniform(2, 4) * samples_per_cycle)
        
        # Ensure duration doesn't exceed waveform length
        max_duration = len(waveform) // 2  # Maximum 50% of waveform length
        duration_samples = min(duration_samples, max_duration)
        
        # Calculate valid start index range
        valid_range = len(waveform) - duration_samples
        if valid_range <= 0:
            valid_range = 1
        
        start_idx = np.random.randint(0, valid_range)
        
        if fault_type == 'sag':
            # More severe voltage sag (20-50% of nominal)
            sag_magnitude = 0.5 + (magnitude * 0.3)  # Results in 0.5-0.8 reduction
            result[start_idx:start_idx + duration_samples] *= (1 - sag_magnitude)
            
            # Add transition ramp for more realistic sag
            ramp_samples = min(100, duration_samples // 10)
            ramp_in = np.linspace(1, 1-sag_magnitude, ramp_samples)
            ramp_out = np.linspace(1-sag_magnitude, 1, ramp_samples)
            result[start_idx:start_idx + ramp_samples] *= ramp_in
            result[start_idx + duration_samples - ramp_samples:start_idx + duration_samples] *= ramp_out
            
        elif fault_type == 'swell':
            # More pronounced voltage swell (120-150% of nominal)
            swell_magnitude = 0.2 + (magnitude * 0.3)  # Results in 1.2-1.5 increase
            result[start_idx:start_idx + duration_samples] *= (1 + swell_magnitude)
            
            # Add transition ramp for more realistic swell
            ramp_samples = min(100, duration_samples // 10)
            ramp_in = np.linspace(1, 1+swell_magnitude, ramp_samples)
            ramp_out = np.linspace(1+swell_magnitude, 1, ramp_samples)
            result[start_idx:start_idx + ramp_samples] *= ramp_in
            result[start_idx + duration_samples - ramp_samples:start_idx + duration_samples] *= ramp_out
            
        elif fault_type == 'interruption':
            # Near-complete interruption (0-5% of nominal)
            int_magnitude = 0.95 + (magnitude * 0.04)  # Results in 0.01-0.05 remaining
            result[start_idx:start_idx + duration_samples] *= (1 - int_magnitude)
            
            # Add transition ramp for more realistic interruption
            ramp_samples = min(50, duration_samples // 20)  # Faster transition for interruption
            ramp_in = np.linspace(1, 1-int_magnitude, ramp_samples)
            ramp_out = np.linspace(1-int_magnitude, 1, ramp_samples)
            result[start_idx:start_idx + ramp_samples] *= ramp_in
            result[start_idx + duration_samples - ramp_samples:start_idx + duration_samples] *= ramp_out
            
        elif fault_type == 'harmonic':
            # Add multiple harmonics with varying amplitudes
            t = np.linspace(0, duration_samples/self.sampling_rate, duration_samples, endpoint=False)
            
            # 3rd harmonic (most prominent)
            h3_magnitude = 0.3 + (magnitude * 0.4)  # 30-70% of fundamental
            result[start_idx:start_idx + duration_samples] += h3_magnitude * np.sin(2 * np.pi * 180 * t)
            
            # 5th harmonic
            h5_magnitude = 0.15 + (magnitude * 0.2)  # 15-35% of fundamental
            result[start_idx:start_idx + duration_samples] += h5_magnitude * np.sin(2 * np.pi * 300 * t)
            
            # 7th harmonic
            h7_magnitude = 0.1 + (magnitude * 0.1)  # 10-20% of fundamental
            result[start_idx:start_idx + duration_samples] += h7_magnitude * np.sin(2 * np.pi * 420 * t)
            
            # Add transition ramp for harmonic content
            ramp_samples = min(100, duration_samples // 10)
            ramp_in = np.linspace(0, 1, ramp_samples)
            ramp_out = np.linspace(1, 0, ramp_samples)
            harmonic_mask = np.ones(duration_samples)
            harmonic_mask[:ramp_samples] = ramp_in
            harmonic_mask[-ramp_samples:] = ramp_out
            
            # Apply the ramp to the added harmonics
            result[start_idx:start_idx + duration_samples] = (
                waveform[start_idx:start_idx + duration_samples] +
                (result[start_idx:start_idx + duration_samples] - 
                 waveform[start_idx:start_idx + duration_samples]) * harmonic_mask
            )
            
        return result

    def generate_batch(self, batch_size=32, include_faults=True):
        """
        Generate a batch of power system waveforms with optional faults.
        
        Args:
            batch_size (int): Number of waveforms to generate
            include_faults (bool): Whether to inject faults into some waveforms
            
        Returns:
            tuple: (waveforms, labels) where labels indicate fault presence and type
        """
        waveforms = []
        labels = []
        fault_types = ['normal', 'sag', 'swell', 'interruption', 'harmonic']
        
        for _ in range(batch_size):
            # Generate base waveform with random phase
            wave = self.generate_base_waveform(phase=np.random.uniform(0, 2*np.pi))
            
            # Add some harmonics
            harmonics = {
                3: np.random.uniform(0, 0.05),  # Reduced background harmonics
                5: np.random.uniform(0, 0.03),
                7: np.random.uniform(0, 0.02)
            }
            wave = self.add_harmonics(wave, harmonics)
            
            # Add noise
            wave = self.add_noise(wave, snr_db=np.random.uniform(35, 45))  # Reduced noise range
            
            if include_faults and np.random.random() < 0.3:  # 30% chance of fault
                fault_type = np.random.choice(fault_types[1:])
                # Calculate fault duration based on cycles
                samples_per_cycle = self.sampling_rate / 60
                duration_samples = int(np.random.uniform(2, 4) * samples_per_cycle)  # Longer faults
                
                wave = self.inject_fault(
                    wave,
                    fault_type,
                    magnitude=np.random.uniform(0.4, 0.8),  # More severe faults
                    duration_samples=duration_samples
                )
                label = fault_types.index(fault_type)
            else:
                label = 0  # normal condition
            
            waveforms.append(wave)
            labels.append(label)
        
        return np.array(waveforms), np.array(labels)

class DatasetGenerator:
    def __init__(self, augmentor, sequence_length=500, stride=100):
        """
        Initialize the dataset generator with appropriate sequence length.
        
        Args:
            augmentor (PowerSystemAugmentor): Instance of PowerSystemAugmentor
            sequence_length (int): Length of sequences for time series data
            stride (int): Steps between sequences
        """
        self.augmentor = augmentor
        self.sequence_length = sequence_length
        self.stride = stride
        
    def generate_training_data(self, n_samples, validation_split=0.2):
        """
        Generate a complete training dataset with validation split.
        
        Args:
            n_samples (int): Total number of samples to generate
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tuple: (train_dataset, val_dataset) as Keras Dataset objects
        """
        n_train = int(n_samples * (1 - validation_split))
        
        # Generate raw waveforms
        raw_x, raw_y = self.augmentor.generate_batch(n_samples)
        
        # Split into train and validation
        x_train = raw_x[:n_train]
        y_train = raw_y[:n_train]
        x_val = raw_x[n_train:]
        y_val = raw_y[n_train:]
        
        # Create time series datasets
        train_dataset = timeseries_dataset_from_array(
            x_train,
            y_train,
            sequence_length=self.sequence_length,
            batch_size=32,
            sequence_stride=self.stride,
            shuffle=True
        )
        
        val_dataset = timeseries_dataset_from_array(
            x_val,
            y_val,
            sequence_length=self.sequence_length,
            batch_size=32,
            sequence_stride=self.stride,
            shuffle=False
        )
        
        return train_dataset, val_dataset