import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import onnxruntime as ort
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import onnxruntime as ort
from matplotlib.animation import FuncAnimation

class WaveformVisualizer:
    def __init__(self):
        self.session = ort.InferenceSession("power_grid_model.onnx")
        self.input_name = self.session.get_inputs()[0].name
        
        # Setup parameters
        self.sample_rate = 6000
        self.base_freq = 60
        self.buffer_size = 100
        self.t = 0
        self.buffer = []
        self.active_fault = None
        self.window_size = 0.05
        
        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)
        
        # Create buttons with proper callbacks
        self.buttons = []  # Keep references to buttons
        labels = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption']
        for i, label in enumerate(labels):
            ax = plt.axes([0.1 + i*0.17, 0.05, 0.15, 0.075])
            btn = Button(ax, label)
            # Create a separate callback function for each button
            btn.on_clicked(self.create_callback(label))
            self.buttons.append(btn)
        
        self.line, = self.ax.plot([], [], 'b-', lw=2)
        self.sample_lines = []
        
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True)
        
        self.detection_text = self.ax.text(0.02, 1.5, '', fontsize=12)
        self.display_buffer = []  # New buffer just for display
        self.detection_buffer = []  # Buffer for ML detection

    def create_callback(self, fault_type):
        """Create a callback function for each button"""
        def callback(event):
            print(f"Setting fault to: {fault_type}")  # Debug print
            self.active_fault = fault_type.lower() if fault_type != 'Normal' else None
        return callback

    def set_fault(self, fault_type):
        self.active_fault = fault_type.lower() if fault_type != 'Normal' else None
        
    def generate_sample(self):
        # Basic sine wave
        sample = np.sin(2 * np.pi * self.base_freq * self.t)
        
        # Apply faults
        if self.active_fault:
            if self.active_fault == 'sag':
                sample *= 0.6  # 60% of normal
            elif self.active_fault == 'swell':
                sample *= 1.4  # 140% of normal
            elif self.active_fault == 'interruption':
                sample *= 0.1  # 10% of normal
            elif self.active_fault == 'harmonic':
                h3 = 0.15 * np.sin(2 * np.pi * 180 * self.t)
                h5 = 0.08 * np.sin(2 * np.pi * 300 * self.t)
                sample = sample + h3 + h5
        
        # Add minimal noise (0.5%)
        sample += np.random.normal(0, 0.005)
        
        return sample

    def update(self, frame):
    # Generate new sample
        sample = self.generate_sample()
        
        # Add to both buffers
        self.display_buffer.append(sample)
        self.detection_buffer.append(sample)
        self.t += 1/self.sample_rate
        
        # Update x-axis limits to move with the signal
        self.ax.set_xlim(self.t - self.window_size, self.t)
        
        # Update plot data using display buffer
        t_data = np.linspace(self.t - len(self.display_buffer)/self.sample_rate, 
                            self.t, len(self.display_buffer))
        self.line.set_data(t_data, self.display_buffer)
        
        # Trim display buffer to window size while keeping continuous waveform
        if len(self.display_buffer) > int(self.window_size * self.sample_rate):
            self.display_buffer = self.display_buffer[-int(self.window_size * self.sample_rate):]
        
        # Update sample markers
        for line in self.sample_lines:
            line.remove()
        self.sample_lines.clear()
        
        # Only run inference when we have exactly bufferSize samples
        if len(self.detection_buffer) == self.buffer_size:
            input_data = np.array([self.detection_buffer], dtype=np.float32)
            output = self.session.run(None, {self.input_name: input_data})
            prediction = ['normal', 'sag', 'swell', 'harmonic', 'interruption'][np.argmax(output[0][0])]
            confidence = np.max(output[0][0])
            
            self.detection_text.set_position((self.t - self.window_size + 0.005, 1.5))
            self.detection_text.set_text(f'Detected: {prediction}\nConfidence: {confidence:.2f}')
            
            # Clear only detection buffer to start collecting next window
            self.detection_buffer = []
            
            # Show window boundary
            self.sample_lines = [self.ax.axvline(self.t, color='r', linestyle=':', alpha=0.5)]
        
        return self.line, self.detection_text

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=20, blit=True)
        plt.show()

if __name__ == "__main__":
    viz = WaveformVisualizer()
    viz.run()