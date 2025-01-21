# Sensor Anomaly Detection Suite (SADS)
## A Power Grid Fault Detection & Monitoring System

## Development Journey Overview 
This project emerged from a 5-day intensive development sprint focused on creating a real-time power system monitoring and fault detection system. What began as an ambitious multi-model approach evolved into a streamlined, production-ready solution that prioritizes reliability and real-world performance.

### 🎯 Project Goals
- Create a browser-based demonstration system for power grid monitoring
- Implement real-time fault detection using modern ML techniques
- Develop an intuitive interface for power system analysis
- Enable interactive fault injection for testing and validation
- Provide real-time visualization of system behavior

## Technical Evolution

### Day 1: Initial Exploration & Architecture
The project began with analyzing power system signals and their characteristics:
- Implemented frequency analysis tools (analyze_frequency.py)
- Developed data validation procedures (test_data_loading.py)
- Created synthetic data generation pipelines (generate_power_system_data.py)
- Established initial ML architecture plans

Key Technologies:
- Python with NumPy/SciPy for signal analysis
- Keras data augmentation for synthetic data
- React for frontend architecture planning

### Day 2: First Implementation Attempt
The initial approach was ambitious, implementing three specialized models:
- Anomaly detection using autoencoders
- Fault classification using CNN
- Waveform pattern detection using YOLO

Technologies Used:
- TensorFlow/Keras for autoencoder implementation
- PyTorch for fault classification
- YOLO for pattern detection
- HDF5 for data management

### Day 3: Data Analysis & Pivot
Visual analysis revealed limitations in the initial approach:
- Confusion matrices showed inconsistent detection patterns
- Multiple models increased complexity and inference time
- Browser deployment challenges with multiple frameworks

Key Insights:
- Need for simplified architecture
- Importance of real-time performance
- Browser compatibility requirements

### Day 4: Final Implementation
The project pivoted to a streamlined approach:
- Single unified model for detection and classification
- Optimized for browser-based inference
- Improved training process with focused data augmentation

Technologies:
- Keras for model development
- ONNX for model conversion
- React hooks for state management
- Web Workers for non-blocking inference

### Day 5: Web Deployment & Optimization
Final implementation and optimization:
- Browser-based inference optimization
- Real-time visualization improvements
- Interactive fault injection system
- Comprehensive validation suite

## Technical Stack

### Frontend
- React 18 with Hooks
- Recharts for real-time visualization
- Web Workers for background processing
- Custom hooks for sensor data management

### Machine Learning
- TensorFlow/Keras for model development
- ONNX Runtime Web for browser inference
- PyTorch for GPU-accelerated model training
- Custom data augmentation pipeline with scikit-learn
- Real-time signal processing

### Development Tools
- Visual Studio Code
- Claude Desktop
- React DevTools
- Git/GitHub
- Chrome DevTools for performance profiling
- Python 3.12+ for ML development

## Key Features

### Real-time Monitoring
- 60Hz power system simulation
- Multi-parameter sensor visualization
- Historical trend analysis
- Performance metrics display

### Fault Detection
- Sag/swell detection
- Harmonic analysis
- Interruption identification
- Real-time classification

### Interactive Testing
- Fault injection interface
- Parameter adjustment
- Real-time response visualization
- Performance validation tools

## Lessons Learned

### Technical Insights
1. Browser-based ML requires careful optimization - the initial multi-model approach proved too heavy for real-time browser inference
2. Real-time visualization demands efficient state management - implemented custom hooks to handle this effectively
3. Web Workers are crucial for smooth UI performance - moved all inference processing off the main thread
4. Single unified models can outperform multiple specialized ones - simplified architecture improved overall performance

### Development Process
1. Early visual validation proved crucial for identifying model training issues
2. Starting with simple architectures would have saved development time
3. Browser compatibility should drive technical decisions from the start
4. Real-world performance trumps theoretical capabilities - the simpler model actually performed better

## Repository Structure

The repository is organized to tell the development story:

* 01_initial_exploration: Signal analysis and validation
* 02_first_approach: Initial multi-model implementation
* 03_data_analysis: Performance analysis and decision points
* 04_final_implementation: Optimized production solution
* 05_web_deployment: Browser deployment and validation

## Contributing

This project was developed in a 5-day sprint as a personal exploration into real-time power system monitoring. Contributions are welcome for:

* Bug fixes
* Performance improvements
* Additional fault patterns
* Enhanced visualizations

## License

MIT License - See LICENSE file for details

## Acknowledgments

Special thanks to:

* The ONNX community for optimization guides
* React ecosystem maintainers
* Various ML communities for architecture insights
* Power system simulation research papers and documentation

---
Developed as a 5-day technical exploration into real-time power system monitoring and fault detection by Todd Wolven. This project demonstrates the evolution from complex multi-model architectures to an efficient, browser-based implementation suitable for real-time monitoring and analysis.