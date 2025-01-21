import tensorflow as tf
from tensorflow import keras
import tf2onnx
import json
import onnx
import numpy as np

def save_model_parameters(model, filename='model_config.json'):
    """Save model parameters and architecture details to a JSON file"""
    config = {
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'layer_details': [],
        'model_name': model.name,
        'optimizer_config': model.optimizer.get_config() if model.optimizer else None,
        'loss_function': model.loss,
        'metrics': model.metrics_names
    }
    
    # Get details of each layer
    for layer in model.layers:
        layer_config = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'units': layer.units if hasattr(layer, 'units') else None,
            'activation': layer.activation.__name__ if hasattr(layer, 'activation') and layer.activation else None,
            'dropout_rate': layer.rate if isinstance(layer, keras.layers.Dropout) else None,
            'trainable_params': layer.count_params()
        }
        config['layer_details'].append(layer_config)

    # Updated preprocessing details with actual performance metrics
    config['preprocessing'] = {
        'input_range': [-1, 1],
        'sampling_rate': 60,
        'sequence_length': 100,
        'signal_conditions': ['normal', 'sag', 'swell', 'harmonic', 'interruption'],
        'pattern_types': ['complete', 'start', 'middle', 'end'],
        'noise_level': 0.005,  # 0.5% noise
        'performance_metrics': {
            'accuracy': 0.9996,  # Updated from actual results
            'normal_precision': 1.00,
            'sag_precision': 1.00,
            'swell_precision': 1.00,
            'harmonic_precision': 1.00,
            'interruption_precision': 1.00
        }
    }

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Model configuration saved to {filename}")

def convert_to_onnx(keras_model_path, onnx_model_path):
    """Convert Keras model to ONNX format"""
    # Load the Keras model
    model = keras.models.load_model(keras_model_path)
    
    # Save model parameters
    save_model_parameters(model)
    
    # Create a sample input for conversion
    spec = (tf.TensorSpec((None, 100), tf.float32, name="input"),)
    
    # Convert to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    # Save ONNX model
    onnx.save_model(model_proto, onnx_model_path)
    
    print(f"Model converted and saved to {onnx_model_path}")
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")

    return model_proto

def verify_onnx_model(onnx_model_path, sample_input_path):
    """Verify ONNX model with sample input"""
    import onnxruntime as ort
    
    # Load sample input
    with open(sample_input_path, 'r') as f:
        sample_data = json.load(f)
    
    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_model_path)
    
    # Prepare input
    input_name = session.get_inputs()[0].name
    input_data = np.array([sample_data['input']], dtype=np.float32)
    
    # Run inference
    output = session.run(None, {input_name: input_data})
    
    # Updated output interpretation for all pattern types
    conditions = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption']
    prediction = output[0][0]
    predicted_class = conditions[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    print("\nONNX Model Verification Results:")
    print("--------------------------------")
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nProbabilities for each class:")
    for cond, prob in zip(conditions, prediction):
        print(f"{cond:>12}: {prob:.4f}")

if __name__ == "__main__":
    # Paths
    keras_model_path = 'power_grid_model.keras'
    onnx_model_path = 'power_grid_model.onnx'
    sample_input_path = 'sample_input.json'
    
    try:
        print("\nStarting model conversion process...")
        # Convert model
        convert_to_onnx(keras_model_path, onnx_model_path)
        
        print("\nVerifying converted model...")
        # Verify conversion
        verify_onnx_model(onnx_model_path, sample_input_path)
        
        print("\nConversion and verification completed successfully!")
        
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")