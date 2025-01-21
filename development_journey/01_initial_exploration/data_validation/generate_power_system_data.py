"""
Power system fault generation with realistic characteristics.
"""

import numpy as np
from scipy import signal
import h5py
from pathlib import Path

class PowerSystemFault:
    def __init__(self, sampling_rate=6000, window_size=1000):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.base_frequency = 60.0
        
    def generate_cycle(self, frequency=60.0, amplitude=1.0, phase=0.0, num_samples=100):
        """Generate a single cycle of a sinusoid."""
        t = np.linspace(0, 1/frequency, num_samples)
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    def create_base_signal(self):
        """Create a pure 60Hz signal."""
        samples_per_cycle = self.sampling_rate / self.base_frequency
        num_cycles = self.window_size / samples_per_cycle
        t = np.linspace(0, num_cycles/self.base_frequency, self.window_size)
        return np.sin(2 * np.pi * self.base_frequency * t)
    
    def add_harmonics(self, signal_data, harmonics_dict):
        """Add specific harmonic content."""
        t = np.linspace(0, len(signal_data)/self.sampling_rate, len(signal_data))
        result = signal_data.copy()
        for harmonic, amplitude in harmonics_dict.items():
            result += amplitude * np.sin(2 * np.pi * harmonic * self.base_frequency * t)
        return result
    
    def apply_rms_envelope(self, signal_data, envelope):
        """Apply an RMS envelope to the signal."""
        # Interpolate envelope to match signal length
        t = np.linspace(0, 1, len(envelope))
        t_new = np.linspace(0, 1, len(signal_data))
        envelope_interp = np.interp(t_new, t, envelope)
        return signal_data * envelope_interp
    
    def generate_normal(self):
        """Generate a normal power system waveform."""
        signal_data = self.create_base_signal()
        # Add minimal background harmonics
        harmonics = {
            3: 0.02,
            5: 0.01,
            7: 0.005
        }
        signal_data = self.add_harmonics(signal_data, harmonics)
        # Add small random variations
        envelope = np.ones(100) + np.random.normal(0, 0.01, 100)
        envelope = signal.savgol_filter(envelope, 11, 3)  # Smooth the variations
        return self.apply_rms_envelope(signal_data, envelope)
    
    def generate_sag(self):
        """Generate a voltage sag."""
        signal_data = self.create_base_signal()
        # Create sag envelope
        envelope = np.ones(100)
        sag_magnitude = np.random.uniform(0.5, 0.8)  # 50-80% voltage remaining
        sag_start = np.random.randint(20, 40)
        sag_end = np.random.randint(60, 80)
        envelope[sag_start:sag_end] = sag_magnitude
        # Smooth transitions
        envelope = signal.savgol_filter(envelope, 11, 3)
        return self.apply_rms_envelope(signal_data, envelope)
    
    def generate_swell(self):
        """Generate a voltage swell."""
        signal_data = self.create_base_signal()
        # Create swell envelope
        envelope = np.ones(100)
        swell_magnitude = np.random.uniform(1.2, 1.5)  # 120-150% voltage
        swell_start = np.random.randint(20, 40)
        swell_end = np.random.randint(60, 80)
        envelope[swell_start:swell_end] = swell_magnitude
        # Smooth transitions
        envelope = signal.savgol_filter(envelope, 11, 3)
        return self.apply_rms_envelope(signal_data, envelope)
    
    def generate_interruption(self):
        """Generate a power interruption."""
        signal_data = self.create_base_signal()
        # Create interruption envelope
        envelope = np.ones(100)
        int_magnitude = np.random.uniform(0, 0.1)  # 0-10% voltage remaining
        int_start = np.random.randint(20, 40)
        int_end = np.random.randint(60, 80)
        envelope[int_start:int_end] = int_magnitude
        # Sharp transitions for interruption
        envelope = signal.savgol_filter(envelope, 5, 3)
        return self.apply_rms_envelope(signal_data, envelope)
    
    def generate_harmonic_distortion(self):
        """Generate harmonic distortion."""
        signal_data = self.create_base_signal()
        # Add significant harmonic content
        harmonics = {
            3: np.random.uniform(0.15, 0.25),
            5: np.random.uniform(0.1, 0.15),
            7: np.random.uniform(0.05, 0.1),
            11: np.random.uniform(0.02, 0.05),
            13: np.random.uniform(0.01, 0.03)
        }
        return self.add_harmonics(signal_data, harmonics)
    
    def generate_sample(self, fault_type=None):
        """Generate a sample with specified fault type."""
        if fault_type is None:
            fault_type = np.random.choice(['normal', 'sag', 'swell', 'interruption', 'harmonic'])
            
        if fault_type == 'normal':
            signal_data = self.generate_normal()
        elif fault_type == 'sag':
            signal_data = self.generate_sag()
        elif fault_type == 'swell':
            signal_data = self.generate_swell()
        elif fault_type == 'interruption':
            signal_data = self.generate_interruption()
        else:  # harmonic
            signal_data = self.generate_harmonic_distortion()
            
        # Add small noise
        noise = np.random.normal(0, 0.01, len(signal_data))
        return signal_data + noise

def generate_dataset(n_samples=8000, output_path='fault_classification.h5'):
    """Generate a complete dataset."""
    generator = PowerSystemFault()
    fault_types = ['normal', 'sag', 'swell', 'interruption', 'harmonic']
    samples_per_type = n_samples // len(fault_types)
    
    x_data = []
    y_data = []
    
    for i, fault_type in enumerate(fault_types):
        print(f"Generating {fault_type} samples...")
        for _ in range(samples_per_type):
            sample = generator.generate_sample(fault_type)
            x_data.append(sample)
            y_data.append(i)
    
    # Convert to arrays and shuffle
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    shuffle_idx = np.random.permutation(len(x_data))
    x_data = x_data[shuffle_idx]
    y_data = y_data[shuffle_idx]
    
    # Split into train/val
    split_idx = int(0.8 * len(x_data))
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    
    # Save to file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('train/data', data=x_train)
        f.create_dataset('train/labels', data=y_train)
        f.create_dataset('val/data', data=x_val)
        f.create_dataset('val/labels', data=y_val)
        
        # Save fault types
        dt = h5py.special_dtype(vlen=str)
        fault_types_ds = f.create_dataset('fault_types', (len(fault_types),), dtype=dt)
        fault_types_ds[:] = fault_types

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = Path("ml/training/data/pytorch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    output_path = output_dir / "fault_classification.h5"
    print(f"Generating dataset at {output_path}...")
    generate_dataset(n_samples=8000, output_path=output_path)
    print("Dataset generation complete!")