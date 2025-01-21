import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def add_noise_variations(signal, base_noise_level=0.005):  # 0.5% instead of 5%
    """Add multiple types of noise to the signal"""
    # White noise (measurement noise)
    white_noise = np.random.normal(0, base_noise_level, len(signal))
    
    # Impulse noise (switching events)
    impulse_noise = np.random.choice([0, 1], len(signal), p=[0.99, 0.01]) * \
                   np.random.normal(0, base_noise_level*2, len(signal))
    
    # Frequency-dependent noise (background harmonics)
    freq_noise = np.sin(np.linspace(0, 10*np.pi, len(signal))) * \
                np.random.normal(0, base_noise_level/4, len(signal))
    
    return signal + white_noise + impulse_noise + freq_noise

def generate_training_data(n_samples=50000):  # Increased sample size significantly
    """Generate synthetic power grid data including partial and transition states"""
    t = np.linspace(0, 1/60, 100)
    
    X = []
    y = []
    labels = []
    
    for _ in range(n_samples):
        base_signal = np.sin(2 * np.pi * 60 * t + np.random.uniform(0, 2*np.pi))
        
        # Generate both complete and partial patterns
        pattern_type = np.random.choice(['complete', 'start', 'middle', 'end'])
        condition = np.random.choice(['normal', 'sag', 'swell', 'harmonic', 'interruption'])
        
        if condition == 'normal':
            signal = base_signal
            label = [1, 0, 0, 0, 0]
            
        else:
            if condition == 'sag':
                depth = np.random.uniform(0.5, 0.8)
                if pattern_type == 'complete':
                    # Full Normal->Sag->Normal
                    duration = np.random.uniform(0.3, 0.7)
                    mask = np.ones(len(t))
                    start = int(len(t) * np.random.uniform(0, 1-duration))
                    end = start + int(len(t) * duration)
                    mask[start:end] = depth
                elif pattern_type == 'start':
                    # Just Normal->Sag
                    mask = np.ones(len(t))
                    start = int(len(t) * 0.5)  # Start in middle
                    mask[start:] = depth
                elif pattern_type == 'middle':
                    # All Sag
                    mask = np.ones(len(t)) * depth
                else:  # end
                    # Just Sag->Normal
                    mask = np.ones(len(t)) * depth
                    end = int(len(t) * 0.5)  # End in middle
                    mask[end:] = 1
                signal = base_signal * mask
                label = [0, 1, 0, 0, 0]
                
            elif condition == 'swell':
                magnitude = np.random.uniform(1.2, 1.5)
                if pattern_type == 'complete':
                    duration = np.random.uniform(0.3, 0.7)
                    mask = np.ones(len(t))
                    start = int(len(t) * np.random.uniform(0, 1-duration))
                    end = start + int(len(t) * duration)
                    mask[start:end] = magnitude
                elif pattern_type == 'start':
                    mask = np.ones(len(t))
                    start = int(len(t) * 0.5)
                    mask[start:] = magnitude
                elif pattern_type == 'middle':
                    mask = np.ones(len(t)) * magnitude
                else:  # end
                    mask = np.ones(len(t)) * magnitude
                    end = int(len(t) * 0.5)
                    mask[end:] = 1
                signal = base_signal * mask
                label = [0, 0, 1, 0, 0]
                
            elif condition == 'harmonic':
                h3 = np.sin(2 * np.pi * 180 * t) * np.random.uniform(0.1, 0.2)
                h5 = np.sin(2 * np.pi * 300 * t) * np.random.uniform(0.05, 0.1)
                h7 = np.sin(2 * np.pi * 420 * t) * np.random.uniform(0.02, 0.05)
                if pattern_type == 'complete':
                    # Gradually introduce harmonics
                    ramp = np.zeros(len(t))
                    start = int(len(t) * 0.3)
                    end = int(len(t) * 0.7)
                    ramp[start:end] = np.linspace(0, 1, end-start)
                    ramp[end:] = np.linspace(1, 0, len(t)-end)
                    signal = base_signal + (h3 + h5 + h7) * ramp
                elif pattern_type == 'start':
                    ramp = np.zeros(len(t))
                    start = int(len(t) * 0.5)
                    ramp[start:] = np.linspace(0, 1, len(t)-start)
                    signal = base_signal + (h3 + h5 + h7) * ramp
                elif pattern_type == 'middle':
                    signal = base_signal + h3 + h5 + h7
                else:  # end
                    ramp = np.ones(len(t))
                    end = int(len(t) * 0.5)
                    ramp[end:] = np.linspace(1, 0, len(t)-end)
                    signal = base_signal + (h3 + h5 + h7) * ramp
                label = [0, 0, 0, 1, 0]
                
            else:  # interruption
                if pattern_type == 'complete':
                    mask = np.ones(len(t))
                    start = np.random.randint(20, 40)
                    duration = np.random.randint(20, 40)
                    mask[start:start+duration] = np.random.uniform(0, 0.1)
                elif pattern_type == 'start':
                    mask = np.ones(len(t))
                    start = int(len(t) * 0.5)
                    mask[start:] = np.random.uniform(0, 0.1)
                elif pattern_type == 'middle':
                    mask = np.ones(len(t)) * np.random.uniform(0, 0.1)
                else:  # end
                    mask = np.ones(len(t)) * np.random.uniform(0, 0.1)
                    end = int(len(t) * 0.5)
                    mask[end:] = 1
                signal = base_signal * mask
                label = [0, 0, 0, 0, 1]

        # Add noise variations
        signal = add_noise_variations(signal, 0.005)  # Using 0.5% noise
        
        X.append(signal)
        y.append(label)
        labels.append(f"{condition}_{pattern_type}")
    
    return np.array(X), np.array(y), labels

def create_model():
    """Create an enhanced model architecture"""
    inputs = keras.Input(shape=(100,))
    
    # First block
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.Dropout(0.2)(x)
    
    # Second block
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    
    # Output layer (now 5 outputs for 5 classes)
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="power_grid_model")

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption'],
                yticklabels=['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_training_history(history):
    """Plot training history with enhanced visualization"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Generate enhanced dataset with more samples and pattern variations
    X, y, labels = generate_training_data(50000)  # Increased from 12000 to 50000
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and compile model
    model = create_model()
    
    # Adjusted learning rate for larger dataset
    initial_learning_rate = 0.0005  # Reduced from 0.001 for better stability

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Adjusted callbacks for larger dataset
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,  # Increased from 20 for more training time
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=15,  # Increased from 10
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Increased epochs and adjusted batch size
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,  # Increased from 100
        batch_size=64,  # Increased from 32 for faster training
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('power_grid_model.keras')

    # Generate and save sample input
    sample_input = X_train[0].tolist()
    with open('sample_input.json', 'w') as f:
        json.dump({"input": sample_input}, f)

    # Evaluate model
    test_predictions = model.predict(X_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix with pattern types
    plot_confusion_matrix(y_test, test_predictions, labels)
    
    # Generate classification report
    class_names = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption']
    report = classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(test_predictions, axis=1),
        target_names=class_names
    )
    
    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)

    print("Training completed and model saved")
    print("\nClassification Report:")
    print(report)

    # Plot the learning rate changes
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('learning_rate.png')
    plt.close()