"""
Test script for power system data augmentation module.
Tests basic functionality of the augmentor and dataset generator.
"""

import numpy as np
import matplotlib.pyplot as plt
from keras_augmentor import PowerSystemAugmentor, DatasetGenerator

def test_base_waveform():
    print("Testing base waveform generation...")
    augmentor = PowerSystemAugmentor()
    wave = augmentor.generate_base_waveform()
    
    plt.figure(figsize=(10, 4))
    plt.plot(wave)
    plt.title("Base 60Hz Waveform")
    plt.grid(True)
    plt.savefig("base_waveform.png")
    plt.close()
    
    assert len(wave) == 256, "Incorrect waveform length"
    assert np.max(wave) <= 1.0, "Amplitude exceeds 1.0"
    print("Base waveform test passed")

def test_harmonics():
    print("Testing harmonic addition...")
    augmentor = PowerSystemAugmentor()
    base_wave = augmentor.generate_base_waveform()
    harmonics = {3: 0.1, 5: 0.05, 7: 0.03}
    wave_with_harmonics = augmentor.add_harmonics(base_wave, harmonics)
    
    plt.figure(figsize=(10, 4))
    plt.plot(base_wave, label='Base')
    plt.plot(wave_with_harmonics, label='With Harmonics')
    plt.title("Waveform with Harmonics")
    plt.legend()
    plt.grid(True)
    plt.savefig("harmonics.png")
    plt.close()
    
    assert len(wave_with_harmonics) == len(base_wave), "Length changed after adding harmonics"
    print("Harmonics test passed")

def test_noise():
    print("Testing noise injection...")
    augmentor = PowerSystemAugmentor()
    base_wave = augmentor.generate_base_waveform()
    noisy_wave = augmentor.add_noise(base_wave, snr_db=30)
    
    plt.figure(figsize=(10, 4))
    plt.plot(base_wave, label='Base')
    plt.plot(noisy_wave, label='Noisy')
    plt.title("Waveform with Noise")
    plt.legend()
    plt.grid(True)
    plt.savefig("noise.png")
    plt.close()
    
    assert len(noisy_wave) == len(base_wave), "Length changed after adding noise"
    print("Noise test passed")

def test_faults():
    print("Testing fault injection...")
    augmentor = PowerSystemAugmentor()
    base_wave = augmentor.generate_base_waveform()
    
    fault_types = ['sag', 'swell', 'interruption', 'harmonic']
    plt.figure(figsize=(15, 10))
    
    for i, fault_type in enumerate(fault_types, 1):
        faulted_wave = augmentor.inject_fault(base_wave.copy(), fault_type)
        plt.subplot(2, 2, i)
        plt.plot(base_wave, label='Base')
        plt.plot(faulted_wave, label=f'With {fault_type}')
        plt.title(f"{fault_type.capitalize()} Fault")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("faults.png")
    plt.close()
    
    print("Fault injection test passed")

def test_batch_generation():
    print("Testing batch generation...")
    augmentor = PowerSystemAugmentor()
    waves, labels = augmentor.generate_batch(batch_size=4)
    
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(waves[i])
        plt.title(f"Sample {i+1}, Label: {labels[i]}")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("batch_samples.png")
    plt.close()
    
    assert len(waves) == 4, "Incorrect batch size"
    assert len(labels) == 4, "Incorrect number of labels"
    print("Batch generation test passed")

def test_dataset_generator():
    print("Testing dataset generator...")
    augmentor = PowerSystemAugmentor()
    generator = DatasetGenerator(augmentor)
    
    train_ds, val_ds = generator.generate_training_data(n_samples=100)
    
    # Check if datasets are created
    assert train_ds is not None, "Training dataset is None"
    assert val_ds is not None, "Validation dataset is None"
    
    print("Dataset generator test passed")

if __name__ == "__main__":
    print("Starting augmentation module tests...")
    test_base_waveform()
    test_harmonics()
    test_noise()
    test_faults()
    test_batch_generation()
    test_dataset_generator()
    print("All tests completed successfully!")