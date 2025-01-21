// src/components/Dashboard.js
import React, { useEffect, useState, useCallback, useMemo, createContext, useRef } from 'react';
import { SensorChart } from './SensorChart';
import { FaultControls } from './FaultControls';
import { MetricsPanel } from './MetricsPanel';
import { AlertBanner } from './AlertBanner';
import { WaveformGenerator } from '../utils/waveformGenerator';

export const EventLogContext = createContext(null);

export const Dashboard = () => {
  // State management
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({
    faultType: null,
    confidence: 0,
    reliability: 0,
    baselinePrecision: 0,
    sampleRate: 6000,
    bufferSize: 100,
    detectionStats: {
      truePositives: 0,
      falsePositives: 0,
      averageLatency: null
    }
  });
  const [alert, setAlert] = useState(null);
  const eventLogRef = useRef(null);
  const workerRef = useRef(null);
  
  // Initialize services
  const generator = useMemo(() => 
    new WaveformGenerator(metrics.sampleRate), [metrics.sampleRate]
  );

// Initialize Web Worker
useEffect(() => {
  workerRef.current = new Worker(
      new URL('../workers/InferenceWorker.js', import.meta.url),
      { type: 'module' }
  );
  workerRef.current.onmessage = (e) => {
      const { type, payload } = e.data;
      
      switch (type) {
          case 'INIT_SUCCESS':
              setIsLoading(false);
              setMetrics(prev => ({
                  ...prev,
                  bufferSize: payload.config.inputSize,
                  sampleRate: payload.config.samplingRate
              }));
              break;

          case 'INIT_ERROR':
              setError(payload.message);
              setIsLoading(false);
              break;

          case 'RESULT':
              handleInferenceResult(payload);
              break;

          case 'ERROR':
              console.error('Inference error:', payload.message);
              if (eventLogRef.current) {
                  eventLogRef.current.addEvent({
                      type: 'error',
                      message: `Inference error: ${payload.message}`
                  });
              }
              break;

          default:
              console.warn('Unhandled message type:', type);
              break;
      }
  };

  // Initialize the inference engine
  workerRef.current.postMessage({ type: 'INIT' });

  return () => {
      workerRef.current?.terminate();
  };
}, [handleInferenceResult]);

  // Handle inference results
const handleInferenceResult = useCallback((result) => {
  setMetrics(prev => ({
    ...prev,
    faultType: result.faultType,
    confidence: result.confidence,
    reliability: result.reliability,
    baselinePrecision: result.baselinePrecision,
    detectionStats: result.detectionStats
  }));

  // Only show alert for actual faults with high confidence
  if (result.faultType && 
      result.faultType.toLowerCase() !== 'normal' && 
      result.confidence > 0.8) {
    setAlert({
      type: 'fault',
      title: 'Fault Detected',
      message: `${result.faultType} detected with ${(result.confidence * 100).toFixed(1)}% confidence`,
      details: `Reliability: ${(result.reliability * 100).toFixed(1)}%`
    });
  } else {
    // Clear any existing alert when returning to normal
    setAlert(null);
  }

  // Log event - only log non-normal faults
  if (eventLogRef.current && 
      result.faultType && 
      result.faultType.toLowerCase() !== 'normal') {
    eventLogRef.current.addEvent({
      type: 'fault',
      message: `${result.faultType} - Confidence: ${(result.confidence * 100).toFixed(1)}%, Reliability: ${(result.reliability * 100).toFixed(1)}%`
    });
  }
}, []);

  const handleNewData = useCallback((data) => {
    if (workerRef.current && data.completeWindow) {
        // Only send if we have a complete window and we're not in transition
        if (!data.isTransitioning) {  // We'll add this flag to the data from WaveformGenerator
            console.log('Sending to worker:', {
                hasCompleteWindow: true,
                windowLength: data.completeWindow.length,
                injectedFault: generator.currentFault
            });

            workerRef.current.postMessage({
                type: 'PROCESS_SAMPLE',
                payload: {
                    completeWindow: data.completeWindow,
                    sample: data.sample,
                    injectedFault: generator.currentFault
                }
            });
        }
    }
}, [generator]);

  // Handle fault injection
  const handleFaultInjection = useCallback((faultParams) => {
    generator.setFault(faultParams);
    
    if (eventLogRef.current && faultParams.type !== 'normal') {
        eventLogRef.current.addEvent({
            type: 'info',
            message: `Injected ${faultParams.type} fault`
        });
    }

    // Only inform worker about non-normal faults
    if (workerRef.current && faultParams.type !== 'normal') {
        workerRef.current.postMessage({
            type: 'SET_INJECTED_FAULT',
            payload: { faultInfo: faultParams }
        });
    }
}, [generator]);

  // Loading and error states remain the same...
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-100 text-red-800 rounded-lg">
        <h2 className="text-lg font-bold">Error</h2>
        <p>{error}</p>
      </div>
    );
  }

// Main render
return (
  <EventLogContext.Provider value={eventLogRef}>
    <div className="min-h-screen bg-gray-900 p-8">
      <AlertBanner alert={alert} />
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-start mb-8">
          <div className="w-1/4 text-gray-300">
            Real-time sensor data analysis using machine learning to detect anomalies. Click the fault buttons to simulate various types of signal disturbances.
          </div>
          
          <h1 className="text-3xl font-bold text-white">
            Sensor Anomaly Detection Suite
          </h1>
          
          <div className="w-1/4 text-gray-300 text-right">
            Press F12 and look at the console to see what the ML Model sees!
          </div>
        </div>
        
        <div className="grid grid-cols-4 gap-6">
          {/* Main Waveform - Left Side */}
          <div className="col-span-3">
            <div className="bg-gray-800 rounded-lg shadow-lg p-4" style={{ height: '500px' }}>
              <SensorChart 
                generator={generator}
                onNewData={handleNewData}
                faultType={metrics.faultType}
                confidence={metrics.confidence}
              />
            </div>
          </div>
          
          {/* Controls and Metrics - Right Side */}
          <div className="col-span-1 space-y-6">
            <div className="bg-gray-800 rounded-lg shadow-lg p-4" style={{ height: '240px' }}>
              <h2 className="text-white text-lg font-semibold mb-4">Fault Controls - Click Me!</h2>
              <FaultControls onInjectFault={handleFaultInjection} />
            </div>
            
            <div className="bg-gray-800 rounded-lg shadow-lg p-4" style={{ height: '236px' }}>
              <h2 className="text-white text-lg font-semibold mb-4">Metrics</h2>
              <MetricsPanel metrics={metrics} />
            </div>
          </div>

          {/* Event Log - Full Width Bottom */}
          <div className="col-span-4">
            <EventLog ref={eventLogRef} />
          </div>
        </div>
      </div>
    </div>
  </EventLogContext.Provider>
);
};

// EventLog component in the same file
const EventLog = React.forwardRef((props, ref) => {
  const [events, setEvents] = useState([]);
  const logRef = useRef(null);

  React.useImperativeHandle(ref, () => ({
    addEvent: (event) => {
        setEvents(prev => {
            const now = Date.now();
            const lastSimilarEvent = prev[0];
            if (lastSimilarEvent && 
                lastSimilarEvent.message === event.message && 
                now - new Date(lastSimilarEvent.timestamp).getTime() < 1000) {
                return prev;
            }
            
            return [{
                id: `${Date.now()}-${Math.random()}`,
                timestamp: new Date().toLocaleTimeString(),
                ...event
            }, ...prev].slice(0, 100);
        });
    }
  }));

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-4">
      <h2 className="text-white text-lg font-semibold mb-2">Event Log</h2>
      <div 
        ref={logRef}
        className="h-[200px] overflow-y-auto space-y-2 text-sm"
      >
        {events.map(event => (
          <div 
            key={event.id} 
            className={`p-2 rounded flex justify-between items-center ${
              event.type === 'fault' ? 'bg-yellow-900/30' : 
              event.type === 'anomaly' ? 'bg-red-900/30' : 
              'bg-gray-900/30'
            }`}
          >
            <div className="flex items-center gap-3 text-white">
              <span className="text-gray-300">{event.timestamp}</span>
              <span>{event.message}</span>
            </div>
            <span className="uppercase font-medium text-gray-300">{event.type}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

EventLog.displayName = 'EventLog';