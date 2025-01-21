import onnxruntime as ort
import numpy as np
import time
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def generate_test_signals(n_samples=100):
    """Generate test signals including partial patterns"""
    t = np.linspace(0, 1/60, 100)
    
    test_data = []
    true_labels = []
    pattern_labels = []  # To track what type of pattern we're testing
    
    for _ in range(n_samples):
        phase = np.random.uniform(0, 2*np.pi)
        base_signal = np.sin(2 * np.pi * 60 * t + phase)
        
        conditions = ['normal', 'sag', 'swell', 'harmonic', 'interruption']
        pattern_types = ['complete', 'start', 'middle', 'end']
        
        for condition in conditions:
            for pattern in pattern_types:
                if condition == 'normal' and pattern != 'complete':
                    continue  # Skip partial patterns for normal condition
                
                signal = base_signal.copy()
                
                if condition == 'normal':
                    pass  # Keep original signal
                    
                elif condition == 'sag':
                    depth = np.random.uniform(0.5, 0.8)
                    if pattern == 'complete':
                        duration = np.random.uniform(0.3, 0.7)
                        mask = np.ones(len(t))
                        start = int(len(t) * np.random.uniform(0, 1-duration))
                        end = start + int(len(t) * duration)
                        mask[start:end] = depth
                    elif pattern == 'start':
                        mask = np.ones(len(t))
                        mask[50:] = depth  # Second half is sag
                    elif pattern == 'middle':
                        mask = np.ones(len(t)) * depth  # All sag
                    else:  # end
                        mask = np.ones(len(t)) * depth
                        mask[50:] = 1  # Second half returns to normal
                    signal = signal * mask
                    
                elif condition == 'swell':
                    magnitude = np.random.uniform(1.2, 1.5)
                    if pattern == 'complete':
                        duration = np.random.uniform(0.3, 0.7)
                        mask = np.ones(len(t))
                        start = int(len(t) * np.random.uniform(0, 1-duration))
                        end = start + int(len(t) * duration)
                        mask[start:end] = magnitude
                    elif pattern == 'start':
                        mask = np.ones(len(t))
                        mask[50:] = magnitude
                    elif pattern == 'middle':
                        mask = np.ones(len(t)) * magnitude
                    else:  # end
                        mask = np.ones(len(t)) * magnitude
                        mask[50:] = 1
                    signal = signal * mask
                    
                elif condition == 'harmonic':
                    h3 = np.sin(2 * np.pi * 180 * t) * np.random.uniform(0.1, 0.2)
                    h5 = np.sin(2 * np.pi * 300 * t) * np.random.uniform(0.05, 0.1)
                    h7 = np.sin(2 * np.pi * 420 * t) * np.random.uniform(0.02, 0.05)
                    harmonics = h3 + h5 + h7
                    if pattern == 'complete':
                        ramp = np.zeros(len(t))
                        ramp[30:70] = np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
                    elif pattern == 'start':
                        ramp = np.zeros(len(t))
                        ramp[50:] = np.linspace(0, 1, 50)
                    elif pattern == 'middle':
                        ramp = np.ones(len(t))
                    else:  # end
                        ramp = np.ones(len(t))
                        ramp[50:] = np.linspace(1, 0, 50)
                    signal = signal + harmonics * ramp
                    
                else:  # interruption
                    if pattern == 'complete':
                        mask = np.ones(len(t))
                        start = np.random.randint(20, 40)
                        duration = np.random.randint(20, 40)
                        mask[start:start+duration] = np.random.uniform(0, 0.1)
                    elif pattern == 'start':
                        mask = np.ones(len(t))
                        mask[50:] = np.random.uniform(0, 0.1)
                    elif pattern == 'middle':
                        mask = np.ones(len(t)) * np.random.uniform(0, 0.1)
                    else:  # end
                        mask = np.ones(len(t)) * np.random.uniform(0, 0.1)
                        mask[50:] = 1
                    signal = signal * mask
                
                # Add minimal noise (0.5%)
                noise = np.random.normal(0, 0.005, len(t))
                signal = signal + noise
                
                test_data.append(signal)
                true_labels.append(conditions.index(condition))
                pattern_labels.append(f"{condition}_{pattern}")
    
    return np.array(test_data), np.array(true_labels), pattern_labels

def plot_confusion_matrix(cm, classes, pattern_labels=None, y_test=None, predictions=None):
    """Plot confusion matrix with optional pattern details"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix (All Patterns)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('validation_confusion_matrix.png')
    plt.close()

    # Only create pattern-specific matrices if we have all the required data
    if all(v is not None for v in [pattern_labels, y_test, predictions]):
        for pattern in ['complete', 'start', 'middle', 'end']:
            pattern_indices = [i for i, label in enumerate(pattern_labels) if pattern in label]
            if pattern_indices:
                pattern_y_true = y_test[pattern_indices]
                pattern_y_pred = predictions[pattern_indices]
                pattern_cm = confusion_matrix(pattern_y_true, pattern_y_pred)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(pattern_cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=classes,
                           yticklabels=classes)
                plt.title(f'Confusion Matrix ({pattern.capitalize()} Patterns)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f'validation_confusion_matrix_{pattern}.png')
                plt.close()

def plot_signal_examples(session, input_name, n_examples=3):
    """Plot examples of each fault type and pattern"""
    conditions = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption']
    patterns = ['complete', 'start', 'middle', 'end']
    
    # Increase figure size and adjust margins
    fig, axes = plt.subplots(len(conditions), len(patterns), figsize=(24, 18))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.4)
    t = np.linspace(0, 1/60, 100)
    
    # Generate one example of each condition and pattern
    test_signals, _, _ = generate_test_signals(1)
    current_idx = 0
    
    for i, condition in enumerate(conditions):
        for j, pattern in enumerate(patterns):
            if condition == 'Normal' and pattern != 'complete':
                axes[i, j].remove()  # Remove subplot for normal partial patterns
                continue
                
            signal = test_signals[current_idx]
            
            # Run inference
            input_data = np.array([signal], dtype=np.float32)
            output = session.run(None, {input_name: input_data})
            predicted_class = conditions[np.argmax(output[0][0])]
            confidence = np.max(output[0][0])
            
            # Plot
            ax = axes[i, j]
            ax.plot(t, signal, 'b-', linewidth=1)
            ax.set_ylim(-2, 2)
            ax.grid(True, alpha=0.3)
            
            if j == 0:
                ax.set_ylabel(condition)
            if i == 0:
                ax.set_title(f'{pattern.capitalize()}')
            
            # Add detection result
            color = 'green' if predicted_class == condition else 'red'
            ax.text(0.05, 1.8, f'Detected: {predicted_class}\nConf: {confidence:.2f}', 
                   color=color, fontsize=8)
            
            current_idx += 1
    
    plt.tight_layout()
    plt.savefig('fault_examples_with_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

def validate_model():
    # Load model
    session = ort.InferenceSession("power_grid_model.onnx")
    input_name = session.get_inputs()[0].name
    
    # Generate test data
    print("Generating test data...")
    X_test, y_test, pattern_labels = generate_test_signals(n_samples=200)
    
    # Performance metrics
    times = []
    predictions = []
    
    print("Running inference...")
    for i, signal in enumerate(X_test):
        start = time.time()
        input_data = np.array([signal], dtype=np.float32)
        output = session.run(None, {input_name: input_data})
        end = time.time()
        
        times.append((end - start) * 1000)
        predictions.append(np.argmax(output[0][0]))
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(X_test)} samples")
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    print("\nPerformance Metrics:")
    print(f"Average inference time: {np.mean(times):.2f}ms")
    print(f"Max inference time: {np.max(times):.2f}ms")
    print(f"Min inference time: {np.min(times):.2f}ms")
    print(f"95th percentile: {np.percentile(times, 95):.2f}ms")
    
    # Generate fault examples
    print("\nGenerating fault examples with detection results...")
    plot_signal_examples(session, input_name)
    
    # Model size
    import os
    model_size = os.path.getsize("power_grid_model.onnx") / (1024 * 1024)
    print(f"\nModel size: {model_size:.2f} MB")
    
    # Accuracy metrics
    accuracy = np.mean(predictions == y_test)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Generate confusion matrices
    conditions = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption']
    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, conditions, pattern_labels, y_test, predictions)
    
    # Per-class and per-pattern metrics
    print("\nPer-class Performance:")
    for pattern in ['complete', 'start', 'middle', 'end']:
        pattern_indices = [i for i, label in enumerate(pattern_labels) if pattern in label]
        if pattern_indices:
            print(f"\n{pattern.capitalize()} Pattern Results:")
            pattern_y_true = y_test[pattern_indices]
            pattern_y_pred = predictions[pattern_indices]
            pattern_acc = np.mean(pattern_y_pred == pattern_y_true)
            print(f"Accuracy: {pattern_acc:.4f}")
            
            for i, condition in enumerate(conditions):
                class_mask = pattern_y_true == i
                if np.any(class_mask):
                    class_acc = np.mean(pattern_y_pred[class_mask] == pattern_y_true[class_mask])
                    print(f"{condition}: {class_acc:.4f}")
    
    # Save detailed results
    results = {
        "performance": {
            "avg_inference_time": float(np.mean(times)),
            "max_inference_time": float(np.max(times)),
            "min_inference_time": float(np.min(times)),
            "percentile_95": float(np.percentile(times, 95)),
            "model_size_mb": float(model_size)
        },
        "accuracy": {
            "overall": float(accuracy),
            "per_pattern": {
                pattern: {
                    "overall": float(np.mean(predictions[pattern_indices] == y_test[pattern_indices])),
                    "per_class": {
                        cond: float(np.mean(
                            predictions[pattern_indices][y_test[pattern_indices] == i] == i
                        )) for i, cond in enumerate(conditions)
                        if np.any(y_test[pattern_indices] == i)
                    }
                }
                for pattern in ['complete', 'start', 'middle', 'end']
                for pattern_indices in [[i for i, l in enumerate(pattern_labels) if pattern in l]]
                if pattern_indices
            }
        }
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    validate_model()