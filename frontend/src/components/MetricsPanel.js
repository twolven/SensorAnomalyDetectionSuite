// src/components/MetricsPanel.js
import React from 'react';

export const MetricsPanel = ({ metrics }) => {
  // Find the status with highest confidence from probabilities
  const getHighestConfidenceStatus = () => {
    if (!metrics.probabilities) {
      // If no probabilities, use the existing confidence value
      return { 
        status: metrics.faultType || 'Normal', 
        confidence: metrics.confidence || 0 
      };
    }
    
    const statuses = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Interruption'];
    const highest = statuses.reduce((max, status, index) => {
        const confidence = metrics.probabilities[index];
        return confidence > max.confidence ? 
            { status, confidence } : max;
    }, { status: 'Normal', confidence: 0 });

    return highest;
  };

  const { status, confidence } = getHighestConfidenceStatus();
  const isFault = status.toLowerCase() !== 'normal';

  const MetricCard = ({ label, value, unit, highlight = false, fullWidth = false, error = false }) => (
    <div className={`bg-gray-900/50 rounded-lg p-3 ${fullWidth ? 'col-span-2' : ''}`}>
      <div className="text-gray-400 text-sm mb-1">{label}</div>
      <div className={`text-xl font-semibold ${error ? 'text-red-500' : highlight ? 'text-yellow-400' : 'text-blue-400'}`}>
        {value}
        {unit && <span className="text-gray-400 text-sm ml-1">{unit}</span>}
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-2 gap-3">
      <MetricCard 
        label="Type" 
        value={isFault ? status : 'Normal Operation'} 
        highlight={isFault}
        fullWidth={true}
      />
      <MetricCard 
        label="Status" 
        value={isFault ? 'Fault!' : 'Nominal'}
        error={isFault}
      />
      <MetricCard 
        label="Confidence" 
        value={`${(metrics.confidence * 100).toFixed(1)}%`}
        highlight={metrics.confidence > 0.8}
      />
    </div>
  );
};