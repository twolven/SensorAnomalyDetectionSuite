import React, { useContext } from 'react';
import { EventLogContext } from './Dashboard';

const FAULT_TYPES = [
  {
    label: 'SAG',
    value: 'sag',
    description: 'Temporarily reduced voltage magnitude (brownout)'
  },
  {
    label: 'SWELL',
    value: 'swell',
    description: 'Temporarily increased voltage magnitude'
  },
  {
    label: 'INTERRUPTION',
    value: 'interruption',
    description: 'Complete loss of voltage for a short duration'
  },
  {
    label: 'HARMONIC',
    value: 'harmonic',
    description: 'Distortion from additional frequency components'
  }
];

export const FaultControls = ({ onInjectFault }) => {
  const handleFaultClick = (faultType) => {
    onInjectFault({ 
        type: faultType,
        magnitude: 0.5,
        oneTime: true  // Add this flag
    });
  };
  
  return (
    <div>
      <div className="flex flex-col space-y-2">
        {FAULT_TYPES.map(({ label, value, description }) => (
            <button
            key={value}
            onClick={() => handleFaultClick(value)}
            className="relative p-2 rounded-lg transition-colors group text-sm w-full bg-gray-700 text-gray-300 hover:bg-gray-600"
            title={description}
            >
            {label}
            <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-gray-900 text-white text-xs rounded-lg w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                {description}
                <div className="absolute right-full top-1/2 -translate-y-1/2 -mr-1 w-2 h-2 bg-gray-900 transform rotate-45"></div>
            </div>
            </button>
        ))}
      </div>
    </div>
  );
};