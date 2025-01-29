export class WaveformGenerator {
    constructor() {
        this.sampleRate = 60;
        this.baseFrequency = 60;
        this.baseAmplitude = 1.0;
        this.time = 0;
        this.windowSize = 100;
        
        this.currentFault = null;
        this.faultActive = false;
        this.skipNextWindow = false;
        this.faultSamplesRemaining = 0;
        
        this.displayBuffer = [];
        this.detectionBuffer = [];
    }
  
    generateSample() {
        // Calculate time point for this sample (matching Python's np.linspace)
        const t = (this.detectionBuffer.length / this.windowSize) * (1/60);
        
        // Generate base sine wave exactly like training data
        let sample = Math.sin(2 * Math.PI * 60 * t);
        
        // Apply fault if active
        if (this.currentFault && this.faultActive && this.faultSamplesRemaining > 0) {
            switch (this.currentFault.type) {
                case 'sag':
                    const sagMagnitude = 0.5 + (this.currentFault.magnitude * 0.2);
                    sample *= sagMagnitude;
                    break;
                    
                case 'swell':
                    const swellMagnitude = 1.2 + (this.currentFault.magnitude * 0.2);
                    sample *= swellMagnitude;
                    break;
                    
                    case 'interruption':
                      // Drop to 0-10% of normal amplitude (matching training data)
                      const intMagnitude = 0.01; // Middle of 0-0.1 range from training
                      sample *= intMagnitude; // Severe drop in amplitude
                      break;
                    
                    case 'harmonic':
                      const fundamental = sample; // Keep the base signal
                      const harmonic3 = Math.sin(2 * Math.PI * 180 * t) * 0.15;  
                      const harmonic5 = Math.sin(2 * Math.PI * 300 * t) * 0.075; 
                      const harmonic7 = Math.sin(2 * Math.PI * 420 * t) * 0.035; 
                      sample = fundamental + harmonic3 + harmonic5 + harmonic7;
                      break;
            }
            
            this.faultSamplesRemaining--;
            if (this.faultSamplesRemaining === 0) {
                this.currentFault = null;
                this.faultActive = false;
            }
        }
  
        // Update buffers
        this.displayBuffer.push(sample);
        this.detectionBuffer.push(sample);
  
        if (this.displayBuffer.length > this.sampleRate * 3) {
            this.displayBuffer.shift();
        }
  
        this.time += 1 / (this.sampleRate * this.baseFrequency);
  
        let completeWindow = null;
        if (this.detectionBuffer.length === this.windowSize) {
            if (!this.skipNextWindow) {
                completeWindow = [...this.detectionBuffer];
            }
            this.detectionBuffer = [];
            this.skipNextWindow = false;
        }
  
        return {
            sample,
            completeWindow,
            isTransitioning: this.skipNextWindow
        };
    }
  
    setFault(faultParams) {
      if (!faultParams || faultParams.type === 'normal') {
          this.clearFault();
          return;
      }
      
      this.currentFault = faultParams;
      this.faultActive = true;
      this.skipNextWindow = true;
  
      // Specific durations for each fault type
      switch (faultParams.type) {
          case 'harmonic':
              this.faultSamplesRemaining = Math.floor(this.windowSize * 0.7); 
              break;
          case 'interruption':
              this.faultSamplesRemaining = Math.floor(this.windowSize * 0.85); 
              break;
          case 'swell':
              this.faultSamplesRemaining = Math.floor(this.windowSize * 0.8);  
              break;
          case 'sag':
              this.faultSamplesRemaining = Math.floor(this.windowSize * 0.5); 
              break;
          default:
              this.faultSamplesRemaining = Math.floor(this.windowSize * 0.375);
      }
  }
  
    clearFault() {
        this.currentFault = null;
        this.faultActive = false;
        this.skipNextWindow = true;
        this.faultSamplesRemaining = 0;
    }
  }
