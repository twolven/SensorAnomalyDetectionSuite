// src/utils/InferenceManager.js
import { DataBuffer } from './DataBuffer';

export class InferenceManager {
    constructor(modelLoader, bufferSize = 1000) {
        this.modelLoader = modelLoader;
        this.dataBuffer = new DataBuffer(bufferSize);
        this.lastAnomalyScore = 0;
        this.lastFaultType = null;
    }

    extractFeatures(waveform) {
        // Debug log
        console.log('Processing waveform:', {
            length: waveform.length,
            sampleRange: [Math.min(...waveform), Math.max(...waveform)]
        });

        // Basic signal features
        const rms = Math.sqrt(waveform.reduce((acc, val) => acc + val * val, 0) / waveform.length);
        const peak = Math.max(...waveform.map(Math.abs));
        const crestFactor = peak / (rms || 1);

        let zeroCrossings = 0;
        for (let i = 1; i < waveform.length; i++) {
            if ((waveform[i - 1] < 0 && waveform[i] >= 0) || 
                (waveform[i - 1] >= 0 && waveform[i] < 0)) {
                zeroCrossings++;
            }
        }

        const mean = waveform.reduce((acc, val) => acc + val, 0) / waveform.length;
        const variance = waveform.reduce((acc, val) => acc + (val - mean) ** 2, 0) / waveform.length;
        const std = Math.sqrt(variance);

        // Frequency content (simple approximation)
        const segments = 4;
        const segmentSize = Math.floor(waveform.length / segments);
        const segmentEnergies = [];
        
        for (let i = 0; i < segments; i++) {
            const segment = waveform.slice(i * segmentSize, (i + 1) * segmentSize);
            const energy = segment.reduce((acc, val) => acc + val * val, 0) / segmentSize;
            segmentEnergies.push(energy);
        }

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
            ...segmentEnergies,    // Energy in different segments
        ];

        // Ensure we have exactly 22 features
        while (features.length < 22) {
            features.push(0);
        }

        console.log('Extracted features:', {
            featuresLength: features.length,
            firstFeatures: features.slice(0, 5)
        });

        return features;
    }

    async processNewSample(sample) {
        this.dataBuffer.add(sample);
    
        if (this.dataBuffer.isFull()) {
            const data = this.dataBuffer.getData();
    
            if (data !== null) {
                try {
                    const waveform = Array.from(data);
                    // Send raw waveform to anomaly detector
                    const anomalyScore = await this.modelLoader.detectAnomaly(waveform);
                    console.log('Anomaly score:', anomalyScore);
                    this.lastAnomalyScore = anomalyScore;
    
                    if (anomalyScore > 0.5) {
                        // Extract features for classifiers
                        const features = this.extractFeatures(waveform);
                        const faultType = await this.modelLoader.classifyFault(features);
                        console.log('Detected fault:', faultType);
                        this.lastFaultType = faultType;
                    } else {
                        this.lastFaultType = null;
                    }
                } catch (error) {
                    console.error('Processing error:', error);
                }
            }
        }
    
        return {
            anomalyScore: this.lastAnomalyScore,
            faultType: this.lastFaultType
        };
    }
}