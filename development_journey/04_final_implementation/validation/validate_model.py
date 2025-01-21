import onnxruntime as ort
import numpy as np

def generate_waves():
    test_cases = []
    t = np.linspace(0, 1/60, 100)
    
    # Normal wave
    normal_wave = np.sin(2 * np.pi * 60 * t)
    test_cases.append({
        'name': 'Normal wave',
        'data': normal_wave,
        'expectedClass': 'normal'
    })
    
    # Swell patterns (120-150% amplitude)
    swell_magnitudes = [1.2, 1.35, 1.5]
    for magnitude in swell_magnitudes:
        # Complete swell (middle portion)
        base = np.sin(2 * np.pi * 60 * t)
        mask = np.ones_like(t)
        middle_mask = (t >= 0.3/60) & (t <= 0.7/60)  # Middle 40%
        mask[middle_mask] = magnitude
        complete_swell = base * mask
        
        test_cases.append({
            'name': f'Complete Swell ({magnitude}x)',
            'data': complete_swell,
            'expectedClass': 'swell'
        })
        
        # Middle-only swell
        middle_swell = np.sin(2 * np.pi * 60 * t) * magnitude
        test_cases.append({
            'name': f'Middle-only Swell ({magnitude}x)',
            'data': middle_swell,
            'expectedClass': 'swell'
        })
    
    # Add wave matching React WaveformGenerator.js
    react_wave = np.sin(2 * np.pi * 60 * t)
    noise = (np.random.rand(len(t)) - 0.5) * 2 * 0.005  # 0.5% noise
    react_wave = react_wave + noise
    test_cases.append({
        'name': 'React Generator Wave',
        'data': react_wave,
        'expectedClass': 'normal'
    })
    
    return test_cases

def print_normal_window():
    t = np.linspace(0, 1/60, 100)
    base_signal = np.sin(2 * np.pi * 60 * t)
    noise = np.random.normal(0, 0.005, len(base_signal))
    signal = base_signal + noise
    print("Training Normal Window:")
    print(f"min: {np.min(signal)}")
    print(f"max: {np.max(signal)}")
    print(f"mean: {np.mean(signal)}")
    print("data:", signal.tolist())


def validate_model():
    print("Loading model...")
    session = ort.InferenceSession("public/models/power_grid_model.onnx")
    input_name = session.get_inputs()[0].name
    
    test_cases = generate_waves()
    
    for case in test_cases:
        print(f"\n=== Testing: {case['name']} ===")
        print("Data stats:", {
            'min': np.min(case['data']),
            'max': np.max(case['data']),
            'mean': np.mean(case['data']),
            'first10': case['data'][:10],
            'last10': case['data'][-10:]
        })
        
        # Run inference
        input_data = np.array([case['data']], dtype=np.float32)
        result = session.run(None, {input_name: input_data})
        probs = result[0][0]
        
        classes = ['normal', 'sag', 'swell', 'harmonic', 'interruption']
        predicted_class = classes[np.argmax(probs)]
        confidence = np.max(probs)
        
        print("Probabilities:", {
            "normal": f"{probs[0]:.4f}",
            "sag": f"{probs[1]:.4f}",
            "swell": f"{probs[2]:.4f}",
            "harmonic": f"{probs[3]:.4f}",
            "interruption": f"{probs[4]:.4f}"
        })
        
        print("Results:", {
            'expectedClass': case['expectedClass'],
            'predictedClass': predicted_class,
            'confidence': f"{confidence:.4f}",
            'correct': predicted_class == case['expectedClass']
        })
        
        # Save test case data if prediction is wrong
        if predicted_class != case['expectedClass']:
            print("\nWRONG PREDICTION - Detailed Data:")
            print("Input:", case['data'].tolist())

if __name__ == "__main__":
    validate_model()
    print_normal_window()