// src/utils/PowerGridInference.js
import * as ort from 'onnxruntime-web';

class PowerGridInference {
    constructor() {
        this.session = null;
        this.config = null;
        this.buffer = [];
        this.bufferSize = 100;  // Explicitly set to match sequence_length
        this.overlap = 50;      
        this.initialized = false;
        this.classes = null;
        this.currentInjectedFault = null;
        this.faultDetectionStats = {
            truePositives: 0,
            falsePositives: 0,
            falseNegatives: 0,
            detectionLatency: []
        };
    }

    calculateAverageLatency = () => {
        if (this.faultDetectionStats.detectionLatency.length === 0) return null;
        const sum = this.faultDetectionStats.detectionLatency.reduce((a, b) => a + b, 0);
        return sum / this.faultDetectionStats.detectionLatency.length;
    };

    async initialize() {
        try {
            // Load model and config
            const modelResponse = await fetch(process.env.PUBLIC_URL + '/models/power_grid_model.onnx');
            const modelBuffer = await modelResponse.arrayBuffer();
            
            const configResponse = await fetch(process.env.PUBLIC_URL + '/models/model_config.json');
            this.config = await configResponse.json();

            // Set class properties from config
            this.bufferSize = this.config.input_shape[0];
            this.classes = this.config.preprocessing.signal_conditions;
            this.inputRange = this.config.preprocessing.input_range;
            
            // Initialize ONNX session
            this.session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.initialized = true;
            console.log('Inference engine initialized with config:', {
                inputShape: this.config.input_shape,
                outputShape: this.config.output_shape,
                signalConditions: this.classes,
                samplingRate: this.config.preprocessing.sampling_rate,
                performanceMetrics: this.config.preprocessing.performance_metrics
            });

            return { 
                status: 'success',
                config: {
                    inputSize: this.bufferSize,
                    classes: this.classes,
                    samplingRate: this.config.preprocessing.sampling_rate
                }
            };
        } catch (error) {
            console.error('Initialization error:', error);
            return { status: 'error', message: error.message };
        }
    }

    setInjectedFault(faultInfo) {
        this.currentInjectedFault = faultInfo ? {
            type: faultInfo.type,
            magnitude: faultInfo.magnitude,
            timestamp: Date.now(),
            detected: false
        } : null;

        if (this.currentInjectedFault === null) {
            this.faultDetectionStats.lastDetectionTime = null;
        }
    }

    preprocessData(data) {
        if (data.length !== this.bufferSize) {
            throw new Error(`Data length must be ${this.bufferSize}`);
        }
    
        // Save the data for analysis
        console.log('React ML Input:', JSON.stringify({
            data: Array.from(data),
            stats: {
                min: Math.min(...data),
                max: Math.max(...data),
                mean: data.reduce((a, b) => a + b, 0) / data.length,
                first10: Array.from(data).slice(0, 10),
                last10: Array.from(data).slice(-10)
            }
        }));
    
        return new ort.Tensor(
            'float32',
            new Float32Array(data),
            [1, this.bufferSize]
        );
    }

    async inference(tensor) {
        try {
            const results = await this.session.run({ [this.session.inputNames[0]]: tensor });
            const output = results[this.session.outputNames[0]].data;
            const probabilities = Array.from(output);
            
            // Log raw probabilities for each class
            console.log('Raw inference probabilities:', {
                normal: probabilities[0].toFixed(4),
                sag: probabilities[1].toFixed(4),
                swell: probabilities[2].toFixed(4),
                harmonic: probabilities[3].toFixed(4),
                interruption: probabilities[4].toFixed(4)
            });
    
            const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
            const confidence = probabilities[maxProbIndex];
            
            // Enforce stricter threshold for non-normal states
            if (maxProbIndex !== 0 && confidence < 0.95) {  // If not normal and confidence < 95%
                return {
                    class: 'normal',
                    probabilities: probabilities,
                    confidence: probabilities[0],  // Use normal class probability
                    baselinePrecision: this.config.preprocessing.performance_metrics.normal_precision,
                    reliability: probabilities[0] * this.config.preprocessing.performance_metrics.normal_precision
                };
            }
    
            return {
                class: this.classes[maxProbIndex],
                probabilities: probabilities,
                confidence: confidence,
                baselinePrecision: this.config.preprocessing.performance_metrics[`${this.classes[maxProbIndex].toLowerCase()}_precision`],
                reliability: confidence * this.config.preprocessing.performance_metrics[`${this.classes[maxProbIndex].toLowerCase()}_precision`]
            };
        } catch (error) {
            console.error('Inference error:', error);
            throw error;
        }
    }
    
    analyzeFaultDetection(result) {
        const now = Date.now();
        const enhancedResult = { ...result };
    
        // Only consider detections above threshold
        const CONFIDENCE_THRESHOLD = 0.85;
        
        if (result.confidence < CONFIDENCE_THRESHOLD) {
            enhancedResult.class = 'normal';
            enhancedResult.confidence = result.confidence;
            return enhancedResult;
        }
    
        if (this.currentInjectedFault) {
            const isCorrectDetection = 
                result.class.toLowerCase() === this.currentInjectedFault.type.toLowerCase() &&
                result.confidence >= CONFIDENCE_THRESHOLD;
    
            if (isCorrectDetection && !this.currentInjectedFault.detected) {
                this.currentInjectedFault.detected = true;
                this.faultDetectionStats.truePositives++;
                
                const latency = now - this.currentInjectedFault.timestamp;
                this.faultDetectionStats.detectionLatency.push(latency);
                enhancedResult.detectionLatency = latency;
    
                enhancedResult.injectionValidation = {
                    injectedFault: this.currentInjectedFault.type,
                    detectionCorrect: true,
                    magnitude: this.currentInjectedFault.magnitude,
                    latency,
                    confidence: result.confidence
                };
            } else if (!isCorrectDetection && result.class !== 'normal') {
                this.faultDetectionStats.falsePositives++;
                enhancedResult.injectionValidation = {
                    injectedFault: this.currentInjectedFault.type,
                    detectionCorrect: false,
                    detectedAs: result.class,
                    confidence: result.confidence
                };
            }
        } else if (result.class.toLowerCase() !== 'normal') {
            this.faultDetectionStats.falsePositives++;
            enhancedResult.injectionValidation = {
                injectedFault: null,
                detectionCorrect: false,
                detectedAs: result.class,
                confidence: result.confidence
            };
        }
    
        enhancedResult.detectionStats = {
            truePositives: this.faultDetectionStats.truePositives,
            falsePositives: this.faultDetectionStats.falsePositives,
            averageLatency: this.calculateAverageLatency(),
            confidence: result.confidence
        };
    
        return enhancedResult;
    }

    // In PowerGridInference.js, update the processSample method:
    async processSample(data, injectedFaultInfo = null) {
        if (!this.initialized) {
            throw new Error('PowerGridInference not initialized');
        }
    
        console.log('Processing in PowerGridInference:', {
            dataLength: data.length,
            expectedLength: this.bufferSize,
            hasInjectedFault: !!injectedFaultInfo
        });
    
        // Update fault info if changed
        if (injectedFaultInfo !== null) {
            this.setInjectedFault(injectedFaultInfo);
        }
    
        // If we received a complete window, process it directly
        if (Array.isArray(data) && data.length === this.bufferSize) {
            try {
                const tensor = this.preprocessData(data);
                const result = await this.inference(tensor);
                return this.analyzeFaultDetection(result);
            } catch (error) {
                console.error('Processing error:', error);
                throw error;
            }
        }
    
        return null;
    }

    countZeroCrossings(data) {
        let crossings = 0;
        for (let i = 1; i < data.length; i++) {
            if ((data[i-1] < 0 && data[i] >= 0) || 
                (data[i-1] >= 0 && data[i] < 0)) {
                crossings++;
            }
        }
        return crossings;
    }
}

export default PowerGridInference;