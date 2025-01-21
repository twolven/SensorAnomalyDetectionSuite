import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- Simplified Waveform Generation (like in keras_augmentor.py) ---

class PowerSystemAugmentor:
    def __init__(self, sampling_rate=12000, window_size=256):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.base_freq = 60.0
        self.time_step = 1.0 / self.sampling_rate

    def generate_base_waveform(self, frequency=60.0, amplitude=1.0, phase=0.0):
        """Generate a base power system waveform."""
        t = np.arange(0, self.window_size) * self.time_step
        print(f"First 10 values of t: {t[:10]}")  # Print first 10 values of t
        print(f"self.time_step: {self.time_step}")  # Print time_step
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)

# --- Frequency Analysis (like in validate_fault_data.py) ---

def calculate_dominant_frequency(waveform, sampling_rate):
    """Calculate dominant frequency using Welch's method."""
    freqs, psd = welch(waveform, fs=sampling_rate, nperseg=min(len(waveform), 256))
    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Exclude DC (index 0)
    return freqs[dominant_freq_idx]

# --- Test Parameters ---

sampling_rate = 12000
window_size = 256
n_samples = 10  # We'll generate a few waveforms

# --- Main Test Script ---

if __name__ == "__main__":
    print("=== Frequency Test ===")

    # 1. Create an Augmentor
    augmentor = PowerSystemAugmentor(sampling_rate=sampling_rate, window_size=window_size)

    # 2. Generate Waveforms (only normal for now)
    waveforms = []
    for _ in range(n_samples):
        wave = augmentor.generate_base_waveform(phase=np.random.uniform(0, 2 * np.pi))
        waveforms.append(wave)
    waveforms = np.array(waveforms)

    # 3. Analyze and Report Dominant Frequency
    frequencies = []
    for waveform in waveforms:
        dom_freq = calculate_dominant_frequency(waveform, sampling_rate)
        frequencies.append(dom_freq)
        print(f"Generated waveform with dominant frequency: {dom_freq:.2f} Hz")

    avg_dom_freq = np.mean(frequencies)
    print(f"\nAverage dominant frequency: {avg_dom_freq:.2f} Hz")

    # 4. Plot Waveforms and Spectra (for visual inspection)
    plt.figure(figsize=(12, 6))

    # Time-domain plot
    plt.subplot(2, 1, 1)
    for i in range(n_samples):
        plt.plot(waveforms[i], alpha=0.7, label=f"Waveform {i+1}")
    plt.title("Generated Waveforms (Time Domain)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()

    # Frequency-domain plot
    plt.subplot(2, 1, 2)
    for i in range(n_samples):
        freqs, psd = welch(waveforms[i], fs=sampling_rate, nperseg=min(len(waveforms[i]), 256))
        plt.semilogy(freqs, psd, alpha=0.7, label=f"Waveform {i+1}")
    plt.title("Frequency Spectra")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nFrequency test complete.")