// validate_model.js
const ort = require('onnxruntime-node');

function generateWaves() {
    const testCases = [];
    const t = Array.from({length: 100}, (_, i) => i / (60 * 100));

    // Normal wave
    const normalWave = t.map(t => Math.sin(2 * Math.PI * 60 * t));
    testCases.push({
        name: 'Normal wave',
        data: normalWave,
        expectedClass: 'normal'
    });

    // Swell patterns (120-150% amplitude)
    const swellMagnitudes = [1.2, 1.35, 1.5];
    swellMagnitudes.forEach(magnitude => {
        // Complete swell (middle portion)
        const completeSwell = t.map(t => {
            const base = Math.sin(2 * Math.PI * 60 * t);
            if (t >= 0.3/60 && t <= 0.7/60) { // Middle 40%
                return base * magnitude;
            }
            return base;
        });
        testCases.push({
            name: `Complete Swell (${magnitude}x)`,
            data: completeSwell,
            expectedClass: 'swell'
        });

        // Middle-only swell
        const middleSwell = t.map(t => Math.sin(2 * Math.PI * 60 * t) * magnitude);
        testCases.push({
            name: `Middle-only Swell (${magnitude}x)`,
            data: middleSwell,
            expectedClass: 'swell'
        });
    });

    // Add generated sine from WaveformGenerator.js for comparison
    const reactWave = t.map(t => {
        let sample = Math.sin(2 * Math.PI * 60 * t);
        const noise = (Math.random() - 0.5) * 2 * 0.005; // 0.5% noise
        return sample + noise;
    });
    testCases.push({
        name: 'React Generator Wave',
        data: reactWave,
        expectedClass: 'normal'
    });

    return testCases;
}

async function validateModel() {
    console.log('Loading model...');
    const model = await ort.InferenceSession.create('./public/models/power_grid_model.onnx');
    
    const testCases = generateWaves();
    
    for (const testCase of testCases) {
        console.log(`\n=== Testing: ${testCase.name} ===`);
        console.log('Data stats:', {
            min: Math.min(...testCase.data),
            max: Math.max(...testCase.data),
            mean: testCase.data.reduce((a, b) => a + b, 0) / testCase.data.length,
            first10: testCase.data.slice(0, 10),
            last10: testCase.data.slice(-10)
        });
        
        const tensor = new ort.Tensor('float32', new Float32Array(testCase.data), [1, 100]);
        const result = await model.run({ 'input': tensor });
        const probs = Array.from(result[model.outputNames[0]].data);
        
        const classes = ['normal', 'sag', 'swell', 'harmonic', 'interruption'];
        const predictedClass = classes[probs.indexOf(Math.max(...probs))];
        const confidence = Math.max(...probs);

        console.log('Probabilities:', {
            normal: probs[0].toFixed(4),
            sag: probs[1].toFixed(4),
            swell: probs[2].toFixed(4),
            harmonic: probs[3].toFixed(4),
            interruption: probs[4].toFixed(4)
        });

        console.log('Results:', {
            expectedClass: testCase.expectedClass,
            predictedClass: predictedClass,
            confidence: confidence.toFixed(4),
            correct: predictedClass === testCase.expectedClass
        });
    }
}

validateModel().catch(console.error);