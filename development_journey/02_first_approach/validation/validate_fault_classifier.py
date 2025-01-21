import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import describe
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from augmentation.keras_augmentor import PowerSystemAugmentor, DatasetGenerator  # Import from your module

def analyze_waveform_characteristics(waveform, sampling_rate):
    """Analyze key characteristics of a waveform."""
    stats = describe(waveform)
    
    # Frequency analysis using Welch's method with optimized parameters
    nperseg = sampling_rate // 10  # Use 0.1s segments (6 cycles at 60 Hz)
    freqs, psd = signal.welch(waveform, 
                            fs=sampling_rate,
                            nperseg=nperseg,
                            noverlap=nperseg//2,
                            scaling='spectrum')
    
    # Find dominant frequency, excluding near-DC components
    dc_cutoff_idx = max(1, int(10 * len(freqs) / sampling_rate))  # Ignore below 10 Hz
    dominant_freq_idx = dc_cutoff_idx + np.argmax(psd[dc_cutoff_idx:])
    dominant_freq = freqs[dominant_freq_idx]
    
    # Calculate RMS more accurately
    rms = np.sqrt(np.mean(np.square(waveform)))
    
    return {
        'mean': stats.mean,
        'variance': stats.variance,
        'skewness': stats.skewness,
        'kurtosis': stats.kurtosis,
        'dominant_freq': dominant_freq,
        'peak_to_peak': np.ptp(waveform),
        'rms': rms
    }

def validate_fault_characteristics(data, labels, fault_types, sampling_rate):
    """Validate the characteristics of each fault type."""
    characteristics = {fault_type: [] for fault_type in fault_types}
    
    for i in range(len(data)):
        fault_type = fault_types[labels[i]]
        chars = analyze_waveform_characteristics(data[i], sampling_rate)
        characteristics[fault_type].append(chars)
    
    stats = {}
    for fault_type in fault_types:
        if not characteristics[fault_type]:
            continue
        
        fault_chars = characteristics[fault_type]
        stats[fault_type] = {
            key: {
                'mean': np.mean([x[key] for x in fault_chars]),
                'std': np.std([x[key] for x in fault_chars]),
            }
            for key in fault_chars[0].keys()
        }
    return stats

def plot_fault_comparison(data, labels, fault_types, save_path, sampling_rate):
    """Create comparison plots for different fault types."""
    n_types = len(fault_types)
    fig, axes = plt.subplots(n_types, 2, figsize=(15, 4 * n_types))

    for i, fault_type in enumerate(fault_types):
        fault_data = data[labels == i]
        if len(fault_data) == 0:
            continue

        # Time domain plot (up to 5 samples)
        num_samples_to_plot = min(len(fault_data), 5)
        for j in range(num_samples_to_plot):
            if n_types > 1:
                axes[i, 0].plot(fault_data[j], label=f'Sample {j+1}')
            else:
                axes[0].plot(fault_data[j], label=f'Sample {j + 1}')

        if n_types > 1:
            axes[i, 0].set_title(f'{fault_type} - Time Domain')
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            axes[i, 0].set_xlabel("Samples")
            axes[i, 0].set_ylabel("Amplitude")

            # Frequency domain plot
            freqs, psd = signal.welch(fault_data[0], fs=sampling_rate, nperseg=min(len(fault_data[0]), 256))
            axes[i, 1].semilogy(freqs, psd)
            axes[i, 1].set_title(f'{fault_type} - Frequency Domain')
            axes[i, 1].grid(True)
            axes[i, 1].set_xlabel("Frequency (Hz)")
            axes[i, 1].set_ylabel("Power Spectral Density")
        else:
            axes[0].set_title(f'{fault_type} - Time Domain')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_xlabel("Samples")
            axes[0].set_ylabel("Amplitude")

            # Frequency domain plot
            freqs, psd = signal.welch(fault_data[0], fs=sampling_rate, nperseg=min(len(fault_data[0]), 256))
            axes[1].semilogy(freqs, psd)
            axes[1].set_title(f'{fault_type} - Frequency Domain')
            axes[1].grid(True)
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Power Spectral Density")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def check_fault_separation(data, labels, fault_types, sampling_rate):
    """Check if fault types are distinguishable using PCA and t-SNE."""
    # Calculate waveform characteristics once and reuse
    waveform_characteristics = [analyze_waveform_characteristics(waveform, sampling_rate) for waveform in data]
    features = np.array([list(chars.values()) for chars in waveform_characteristics])

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data) - 1))
    tsne_result = tsne.fit_transform(features)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for i, fault_type in enumerate(fault_types):
        mask = labels == i
        ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], label=fault_type, alpha=0.6)
        ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=fault_type, alpha=0.6)

    ax1.set_title('PCA Visualization')
    ax1.legend()
    ax2.set_title('t-SNE Visualization')
    ax2.legend()
    plt.savefig('fault_separation.png')
    plt.close()

    # Compute separation metrics
    pca_separation = silhouette_score(features, labels)
    tsne_separation = silhouette_score(features, labels)

    return pca_separation, tsne_separation

def verify_fault_consistency(data, labels, fault_types, max_comparisons=1000):
    """Verify that fault characteristics are consistent within classes."""
    consistency_metrics = {}

    for i, fault_type in enumerate(fault_types):
        fault_data = data[labels == i]
        if len(fault_data) < 2:  # Need at least 2 samples to compare
            continue

        n_samples = len(fault_data)
        correlations = []

        # Limit the number of comparisons
        if n_samples * (n_samples - 1) / 2 > max_comparisons:
            # Randomly select pairs for comparison
            np.random.seed(42)  # For reproducibility
            indices = []
            while len(indices) < max_comparisons:
                idx1 = np.random.randint(0, n_samples)
                idx2 = np.random.randint(0, n_samples)
                if idx1 != idx2 and (idx1, idx2) not in indices and (idx2, idx1) not in indices:
                    indices.append((idx1, idx2))
        else:
            # If small enough, do all comparisons
            indices = [(j, k) for j in range(n_samples) for k in range(j + 1, n_samples)]

        # Calculate correlations for selected pairs using np.corrcoef
        for idx1, idx2 in indices:
            corr = np.corrcoef(fault_data[idx1], fault_data[idx2])[0, 1]  # Get the correlation coefficient
            correlations.append(corr)

        consistency_metrics[fault_type] = {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'n_comparisons': len(correlations)
        }

    return consistency_metrics

def main():
    print("=== Validating Fault Data Generation ===")

    # Parameters for data generation - using optimized values for 60Hz
    sampling_rate = 6000  # Optimized for 60Hz (100 samples per cycle)
    window_size = 1000   # 10 cycles at 60Hz
    n_samples = 8000

    # Generate data using your augmentor
    augmentor = PowerSystemAugmentor(sampling_rate=sampling_rate, window_size=window_size)
    data, labels = augmentor.generate_batch(n_samples)
    fault_types = ['normal', 'sag', 'swell', 'interruption', 'harmonic']

    print("\n1. Basic Data Statistics:")
    print(f"Number of samples: {len(data)}")
    print(f"Sample length: {data.shape[1]}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Window size: {window_size} samples")
    print(f"Duration: {window_size/sampling_rate:.3f} seconds")
    print(f"Expected cycles: {(window_size/sampling_rate * 60):.1f}")
    
    print("\nClass distribution:")
    for i, fault_type in enumerate(fault_types):
        count = np.sum(labels == i)
        print(f"{fault_type}: {count} samples ({count/len(data)*100:.1f}%)")

    print("\n2. Analyzing Fault Characteristics...")
    stats = validate_fault_characteristics(data, labels, fault_types, sampling_rate)  # Pass sampling_rate

    print("\nKey characteristics by fault type:")
    for fault_type, characteristics in stats.items():
        print(f"\n{fault_type}:")
        print(f"  RMS: {characteristics['rms']['mean']:.3f} ± {characteristics['rms']['std']:.3f}")
        print(f"  Dominant Frequency: {characteristics['dominant_freq']['mean']:.1f} Hz")
        print(f"  Peak-to-Peak: {characteristics['peak_to_peak']['mean']:.3f}")

    print("\n3. Checking Fault Separation...")
    pca_sep, tsne_sep = check_fault_separation(data, labels, fault_types, sampling_rate)  # Pass sampling_rate
    print(f"PCA Separation Score: {pca_sep:.3f}")
    print(f"t-SNE Separation Score: {tsne_sep:.3f}")

    print("\n4. Verifying Fault Consistency...")
    consistency = verify_fault_consistency(data, labels, fault_types)

    print("\nWithin-class consistency:")
    for fault_type, metrics in consistency.items():
        print(f"\n{fault_type}:")
        print(f"  Mean correlation: {metrics['mean_correlation']:.3f}")
        print(f"  Correlation std: {metrics['std_correlation']:.3f}")

    print("\n5. Generating Visualization Plots...")
    plot_fault_comparison(data, labels, fault_types, 'fault_comparison.png', sampling_rate)  # Pass sampling_rate

    print("\nValidation complete! Check the generated plots for visual analysis.")

if __name__ == "__main__":
    main()