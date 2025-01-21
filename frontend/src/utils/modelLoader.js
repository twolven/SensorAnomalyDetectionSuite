// src/utils/modelLoader.js
import * as ort from 'onnxruntime-web';

export class ModelLoader {
  constructor() {
    this.anomalyDetector = null;
    this.classifiers = {
      harmonic: null,
      interruption: null,
      normal: null,
      sag: null,
      swell: null
    };
    this.threshold = null;
    this.config = null;
  }

  async loadModels() {
    try {
      const publicUrl = process.env.PUBLIC_URL || '';

      // Load classifier configuration first
      const classifierConfigResponse = await fetch(`${publicUrl}/models/classifier_config.json`);
      this.config = await classifierConfigResponse.json();

      // Load anomaly detector
      this.anomalyDetector = await ort.InferenceSession.create(
        `${publicUrl}/models/anomaly_detector.onnx`
      );

      // Load classifiers
      for (const key of Object.keys(this.classifiers)) {
        this.classifiers[key] = await ort.InferenceSession.create(
          `${publicUrl}/models/${key}_classifier.onnx`
        );
      }

      // Load threshold configuration
      const response = await fetch(`${publicUrl}/models/threshold.json`);
      const thresholdConfig = await response.json();
      this.threshold = thresholdConfig.threshold;

      return true;
    } catch (error) {
      console.error('Error loading models:', error);
      return false;
    }
  }

    // src/utils/modelLoader.js
    async detectAnomaly(waveform) {
        if (!this.anomalyDetector || !this.config) return 0;
    
        try {
            // Create the correct size Float32Array first
            const modelInput = new Float32Array(128 * 1000);
            
            // Fill it with the waveform data
            for (let i = 0; i < 128; i++) {
                for (let j = 0; j < waveform.length; j++) {
                    modelInput[i * 1000 + j] = waveform[j];
                }
            }
    
            // Create tensor directly from the Float32Array
            const tensor = new ort.Tensor(
                'float32',
                modelInput,  // Already a Float32Array of correct size
                [1, 128, 1000]
            );
    
            const results = await this.anomalyDetector.run({ input: tensor });
            const outputTensor = results['conv1d_6'];
            
            const values = Array.from(outputTensor.data);
            const score = values.reduce((sum, val) => sum + Math.abs(val), 0) / values.length;
            const normalizedScore = 1 / (1 + Math.exp(-10 * score));
            
            return normalizedScore > this.threshold ? normalizedScore : 0;
        } catch (error) {
            console.error('Anomaly detection error:', error);
            return 0;
        }
    }
    
    async classifyFault(waveform) {
        if (!this.config) return null;
    
        try {
            // Extract features
            const features = this.extractFeatures(waveform);
            
            // Create tensor with shape [1, 22]
            const tensor = new ort.Tensor(
                'float32',
                new Float32Array(features),
                [1, 22]
            );
    
            // Run all classifiers
            const results = await Promise.all(
                Object.entries(this.classifiers).map(async ([type, model]) => {
                    const output = await model.run({ input: tensor });
                    return {
                        type,
                        probability: output.output.data[0]
                    };
                })
            );
    
            // Find highest probability classification
            const bestResult = results.reduce((best, current) => 
                current.probability > best.probability ? current : best
            );
    
            return bestResult.type;
        } catch (error) {
            console.error('Classification error:', error);
            return null;
        }
    }
    
    extractFeatures(waveform) {
        // RMS
        const rms = Math.sqrt(waveform.reduce((acc, val) => acc + val * val, 0) / waveform.length);
        
        // Peak
        const peak = Math.max(...waveform.map(Math.abs));
        
        // Crest factor
        const crestFactor = peak / (rms || 1);
        
        // Zero crossings
        let zeroCrossings = 0;
        for (let i = 1; i < waveform.length; i++) {
            if ((waveform[i-1] < 0 && waveform[i] >= 0) || 
                (waveform[i-1] >= 0 && waveform[i] < 0)) {
                zeroCrossings++;
            }
        }
        
        // Statistical features
        const mean = waveform.reduce((acc, val) => acc + val, 0) / waveform.length;
        const variance = waveform.reduce((acc, val) => acc + (val - mean) ** 2, 0) / waveform.length;
        const std = Math.sqrt(variance);
        
        // Energy in segments
        const segments = 4;
        const segmentSize = Math.floor(waveform.length / segments);
        const segmentEnergies = [];
        
        for (let i = 0; i < segments; i++) {
            const segment = waveform.slice(i * segmentSize, (i + 1) * segmentSize);
            const energy = segment.reduce((acc, val) => acc + val * val, 0) / segmentSize;
            segmentEnergies.push(energy);
        }
        
        // Combine all features
        const features = [
            rms,                    // RMS value
            peak,                   // Peak value
            crestFactor,           // Crest factor
            mean,                  // Mean
            std,                   // Standard deviation
            zeroCrossings / waveform.length, // Normalized zero crossings
            variance,             // Variance
            Math.max(...waveform), // Maximum
            Math.min(...waveform), // Minimum
            peak / (Math.abs(mean) || 1),  // Peak to mean ratio
            ...segmentEnergies,    // Energy in segments
        ];
        
        // Pad to 22 features
        while (features.length < 22) {
            features.push(0);
        }
        
        return features;
    }
}