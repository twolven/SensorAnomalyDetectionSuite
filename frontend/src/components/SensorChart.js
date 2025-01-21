// src/components/SensorChart.js
import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';

const WINDOW_SIZE = 500;  // For visualization
const ML_WINDOW_SIZE = 100;  // For ML detection
const SAMPLE_INTERVAL = Math.floor(1000 / 60);  // 60Hz sampling rate (~16.67ms)

export const SensorChart = ({ generator, onNewData, faultType, confidence }) => {
    console.log('SensorChart mounting with generator:', !!generator);
    const [data, setData] = useState([]);
    const [isRunning, setIsRunning] = useState(true);
    const [errorMessage, setErrorMessage] = useState(null);
    const timeoutRef = useRef(null);
    const timeRef = useRef(0);
    const mlBufferRef = useRef([]);  // Buffer for ML samples
    const windowStateRef = useRef('normal'); // Track window state
    const transitionCountRef = useRef(0); // Track transitions within a window
    
    const updateData = () => {
        console.log('updateData called, isRunning:', isRunning);
        if (!isRunning) return;
    
        try {
            const { sample } = generator.generateSample();
            console.log('Sample generated:', sample);
            timeRef.current += 1 / 60; // 60Hz sampling
            
            // Add to visualization data
            setData(prevData => {
                const newData = [...prevData, {
                    time: timeRef.current,
                    value: sample,
                    threshold: faultType ? confidence : null
                }];
                
                if (newData.length > WINDOW_SIZE) {
                    newData.shift();
                }
                return newData;
            });

            // Add to ML buffer
            mlBufferRef.current.push(sample);

            // Handle window completion
            if (mlBufferRef.current.length === ML_WINDOW_SIZE) {
                console.log('ML window complete:', {
                    windowState: windowStateRef.current,
                    samples: mlBufferRef.current.length
                });

                // Send complete window
                if (onNewData) {
                    onNewData({
                        sample: sample,
                        completeWindow: [...mlBufferRef.current]
                    });
                }

                // Reset buffer and prepare for next window
                mlBufferRef.current = [];
                
                // Update window state for next collection
                if (generator.currentFault && generator.currentFault.type !== 'normal') {
                    // If fault is active, alternate between normal and fault windows
                    windowStateRef.current = windowStateRef.current === 'normal' ? 'fault' : 'normal';
                } else {
                    // If no fault is active, stay in normal state
                    windowStateRef.current = 'normal';
                }

                // Reset transition counter for new window
                transitionCountRef.current = 0;
            }

            // Schedule next update at exact 60Hz
            console.log('Scheduling next update with interval:', SAMPLE_INTERVAL);
            timeoutRef.current = setTimeout(updateData, SAMPLE_INTERVAL);

        } catch (err) {
            console.error('Chart update error:', err);
            setErrorMessage(`Chart update error: ${err.message}`);
            setIsRunning(false);
        }
    };

    // Effect for starting/stopping the updates
    useEffect(() => {
        console.log('Running effect, isRunning:', isRunning);
        if (isRunning) {
            updateData();
        }
    
        return () => {
            console.log('Cleaning up effect');
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, [isRunning]);

    // Reset when generator changes
    useEffect(() => {
        console.log('Generator changed, resetting data');
        setData([]);
        timeRef.current = 0;
        mlBufferRef.current = [];
        windowStateRef.current = 'normal';
        transitionCountRef.current = 0;
    }, [generator, updateData]);

    const toggleRunning = () => {
        setIsRunning(prev => !prev);
        setErrorMessage(null);
    };

    return (
        <div className="bg-gray-900 p-4 rounded-lg shadow-lg">
            <div className="flex justify-between items-center mb-2">
                <button 
                    onClick={toggleRunning}
                    className={`px-4 py-2 rounded ${
                        isRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                    } text-white`}
                >
                    {isRunning ? 'Stop' : 'Start'}
                </button>
                <div className="text-white text-sm">
                    {faultType ? (
                        `Detected: ${faultType} (${(confidence * 100).toFixed(1)}% confidence)`
                    ) : (
                        `Samples: ${data.length}`
                    )}
                </div>
            </div>
            {errorMessage && (
                <div className="text-red-500 text-sm mb-2">
                    Error: {errorMessage}
                </div>
            )}
            <div className="h-[380px] w-full">
                <ResponsiveContainer>
                    <LineChart 
                        data={data} 
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis 
                            dataKey="time" 
                            type="number"
                            domain={['dataMin', 'dataMax']}
                            tickFormatter={(value) => value.toFixed(2)}
                            stroke="#666"
                        />
                        <YAxis 
                            domain={[
                                dataMin => Math.min(-1.5, dataMin * 1.1),
                                dataMax => Math.max(1.5, dataMax * 1.1)
                            ]}
                            stroke="#888"
                            tick={{ fill: '#888' }}
                            tickFormatter={value => value.toFixed(2)}
                        />
                        <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke={faultType ? '#ef4444' : '#3b82f6'} 
                            strokeWidth={2}
                            dot={false} 
                            isAnimationActive={false}
                        />
                        {faultType && (
                            <Line
                                type="monotone"
                                dataKey="threshold"
                                stroke="#ef4444"
                                strokeWidth={1.5}
                                strokeDasharray="5 5"
                                dot={false}
                            />
                        )}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};