// src/utils/ExportData.js
export const exportData = (data, type = 'csv') => {
    switch (type) {
        case 'csv':
            return exportCSV(data);
        case 'json':
            return exportJSON(data);
        default:
            throw new Error('Unsupported export type');
    }
};

function exportCSV(data) {
    const headers = ['timestamp', 'value', 'anomalyScore', 'faultType'];
    const csvContent = [
        headers.join(','),
        ...data.map(row => [
            row.timestamp,
            row.value,
            row.anomalyScore,
            row.faultType || 'normal'
        ].join(','))
    ].join('\n');

    downloadFile(csvContent, 'power_system_data.csv', 'text/csv');
}

function exportJSON(data) {
    const jsonContent = JSON.stringify(data, null, 2);
    downloadFile(jsonContent, 'power_system_data.json', 'application/json');
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
}