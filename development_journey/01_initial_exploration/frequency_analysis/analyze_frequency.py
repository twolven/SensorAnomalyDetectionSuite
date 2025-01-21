import numpy as np
import matplotlib.pyplot as plt

# Adjusted parameters for better 60 Hz capture
sampling_rate = 6000  # Reduced to get better bin alignment
window_size = 1000    # Increased for better resolution
frequency = 60        # Target frequency
duration = window_size / sampling_rate

# Calculate important metrics
samples_per_cycle = sampling_rate / frequency
num_cycles = duration * frequency
freq_resolution = sampling_rate / window_size

print(f"Analysis Parameters:")
print(f"Samples per cycle: {samples_per_cycle:.1f}")
print(f"Number of cycles: {num_cycles:.1f}")
print(f"Frequency resolution: {freq_resolution:.2f} Hz")

# Generate signal
t = np.linspace(0, duration, window_size, endpoint=False)
signal = np.sin(2 * np.pi * frequency * t)

# Compute FFT
fft_values = np.fft.rfft(signal)  # Using rfft for real signals
fft_freqs = np.fft.rfftfreq(window_size, 1/sampling_rate)
magnitude_spectrum = 2.0/window_size * np.abs(fft_values)

# Find dominant frequency
max_freq_idx = np.argmax(magnitude_spectrum[1:]) + 1
dominant_freq = fft_freqs[max_freq_idx]

print(f"\nFFT Analysis:")
print(f"Dominant frequency: {dominant_freq:.2f} Hz")
print(f"\nTop 5 frequency components:")
top_indices = np.argsort(magnitude_spectrum[1:])[-5:] + 1
for idx in reversed(top_indices):
    print(f"  {fft_freqs[idx]:.2f} Hz: {magnitude_spectrum[idx]:.4f}")

# Plotting
plt.figure(figsize=(12, 8))

# Time domain plot
plt.subplot(211)
plt.plot(t[:100] * 1000, signal[:100])  # Show first 100 samples
plt.title('Time Domain Signal (first 100 samples)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)

# Frequency domain plot
plt.subplot(212)
plt.plot(fft_freqs, magnitude_spectrum)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.axvline(x=60, color='r', linestyle='--', label='60 Hz')
max_plot_freq = 180  # Show up to 180 Hz
plt.xlim(0, max_plot_freq)
plt.legend()

plt.tight_layout()
plt.show()

# Verify signal frequency through zero crossings
zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
if len(zero_crossings) >= 4:
    # Calculate frequency from zero crossings
    avg_period = np.mean(np.diff(zero_crossings)) * 2 / sampling_rate  # *2 because we need two crossings for a period
    measured_freq = 1 / avg_period
    print(f"\nFrequency verification through zero crossings: {measured_freq:.2f} Hz")