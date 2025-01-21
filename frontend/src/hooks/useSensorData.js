// src/hooks/useSensorData.js
import { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

const SAMPLE_RATE = 6000; // 6kHz sampling rate
const BUFFER_SIZE = 1000; // 1000 samples buffer

export const useSensorData = (models) => {
  const [data, setData] = useState([]);
  const [faultState, setFaultState] = useState(null);
  const [anomalyScore, setAnomalyScore] = useState(0);
  const timeRef = useRef(0);
  
  // Generate base waveform with optional fault injection
  const generateSample = (t, fault = null) => {
    let value = Math.sin(2 * Math.PI * 60 * t); // Base 60Hz signal
    
    if (fault) {
      switch (fault.type) {
        case 'sag':
          value *= (1 - fault.magnitude * 0.7);
          break;
        case 'swell':
          value *= (1 + fault.magnitude * 0.5);
          break;
        case 'interruption':
          value *= (1 - fault.magnitude * 0.95);
          break;
        case 'harmonic':
          value += fault.magnitude * 0.3 * Math.sin(2 * Math.PI * 180 * t);
          value += fault.magnitude * 0.2 * Math.sin(2 * Math.PI * 300 * t);
          break;
        default:
          break;
      }
    }
    
    return value;
  };

  // Run model inference
  const runInference = async (waveform) => {
    if (!models) return;

    try {
      // Prepare input tensor
      const inputTensor = new ort.Tensor(
        'float32',
        new Float32Array(waveform),
        [1, waveform.length]
      );

      // Run anomaly detection
      const anomalyResult = await models.anomaly.run({ input: inputTensor });
      const anomalyScore = anomalyResult.output.data[0];

      // Run fault classification if anomaly detected
      if (anomalyScore > 0.5) {
        for (const [faultType, model] of Object.entries(models)) {
          if (faultType === 'anomaly') continue;
          const result = await model.run({ input: inputTensor });
          if (result.output.data[0] > 0.8) {
            return { type: faultType, score: result.output.data[0] };
          }
        }
      }

      setAnomalyScore(anomalyScore);
      return null;
    } catch (error) {
      console.error('Inference error:', error);
      return null;
    }
  };

  // Update data and run inference
  useEffect(() => {
    const updateData = async () => {
      timeRef.current += 1 / SAMPLE_RATE;
      const newValue = generateSample(timeRef.current, faultState);
      
      setData(prev => {
        const newData = [...prev, {
          time: timeRef.current,
          value: newValue
        }];
        
        if (newData.length > BUFFER_SIZE) {
          newData.shift();
        }
        
        // Run inference on latest buffer
        const waveform = newData.map(d => d.value);
        runInference(waveform);
        
        return newData;
      });
    };

    const interval = setInterval(updateData, 1000 / 60); // 60 FPS update
    return () => clearInterval(interval);
  }, [faultState, models]);

  return {
    data,
    anomalyScore,
    setFaultState,
  };
};