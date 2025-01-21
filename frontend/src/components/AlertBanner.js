// src/components/AlertBanner.js
import React from 'react';

export const AlertBanner = ({ alert }) => {
    if (!alert) return null;

    const getAlertStyle = (type) => {
        switch (type) {
            case 'fault':
                return 'bg-red-600';
            case 'warning':
                return 'bg-yellow-600';
            case 'info':
                return 'bg-blue-600';
            default:
                return 'bg-gray-600';
        }
    };

    return (
        <div className={`fixed top-0 left-0 right-0 ${getAlertStyle(alert.type)} text-white px-4 py-2 shadow-lg z-50`}>
            <div className="max-w-7xl mx-auto flex justify-between items-center">
                <div className="flex items-center space-x-2">
                    <span className="font-bold">{alert.title}</span>
                    <span>{alert.message}</span>
                </div>
                {alert.details && (
                    <span className="text-sm opacity-75">{alert.details}</span>
                )}
            </div>
        </div>
    );
};