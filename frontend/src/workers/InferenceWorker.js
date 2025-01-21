/* eslint-disable no-restricted-globals */
import PowerGridInference from '../utils/PowerGridInference';

const ctx = self;
let inferenceEngine = null;
let sampleBuffer = [];
const WINDOW_SIZE = 100;

ctx.addEventListener('message', async function(e) {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT':
            try {
                console.log('Starting inference engine initialization...');
                inferenceEngine = new PowerGridInference();
                console.log('PowerGridInference instance created');
                
                const initResult = await inferenceEngine.initialize();
                console.log('Initialization result:', initResult);
                
                if (initResult.status === 'success') {
                    console.log('Inference engine successfully initialized');
                    ctx.postMessage({ 
                        type: 'INIT_SUCCESS',
                        payload: initResult 
                    });
                } else {
                    console.error('Initialization failed:', initResult);
                    ctx.postMessage({ 
                        type: 'INIT_ERROR', 
                        payload: initResult 
                    });
                }
            } catch (error) {
                console.error('Worker initialization error:', {
                    message: error.message,
                    stack: error.stack
                });
                ctx.postMessage({ 
                    type: 'INIT_ERROR', 
                    payload: { 
                        message: error.message,
                        stack: error.stack
                    } 
                });
            }
            break;

            case 'PROCESS_SAMPLE':
              if (!inferenceEngine || !inferenceEngine.initialized) {
                  console.log('Inference engine not ready:', {
                      engineExists: !!inferenceEngine,
                      initialized: inferenceEngine?.initialized
                  });
                  ctx.postMessage({ 
                      type: 'ERROR', 
                      payload: { message: 'Inference engine not ready' } 
                  });
                  return;
              }
          
              try {
                  // If we received a complete window, process it directly
                  if (payload.completeWindow) {
                      console.log('InferenceWorker received complete window:', {
                          windowLength: payload.completeWindow.length,
                          firstSample: payload.completeWindow[0],
                          lastSample: payload.completeWindow[payload.completeWindow.length - 1]
                      });
                      
                      const result = await inferenceEngine.processSample(
                          payload.completeWindow,
                          payload.injectedFault
                      );
                      
                      console.log('Inference result:', result);
                      
                      if (result) {
                          ctx.postMessage({ 
                              type: 'RESULT', 
                              payload: {
                                  faultType: result.class,
                                  confidence: result.confidence,
                                  reliability: result.reliability,
                                  probabilities: result.probabilities,
                                  baselinePrecision: result.baselinePrecision,
                                  injectionValidation: result.injectionValidation,
                                  detectionStats: result.detectionStats
                              }
                          });
                      } else {
                          console.log('No result from inference engine');
                      }
                  } else {
                      console.log('Received sample without complete window');
                  }
              } catch (error) {
                  console.error('Detailed inference error:', {
                      message: error.message,
                      stack: error.stack,
                      payload
                  });
                  
                  ctx.postMessage({ 
                      type: 'ERROR', 
                      payload: { 
                          message: error.message,
                          details: error.stack 
                      } 
                  });
              }
              break;
              
        case 'SET_INJECTED_FAULT':
            if (inferenceEngine) {
                inferenceEngine.setInjectedFault(payload.faultInfo);
                // Clear buffer when fault changes
                sampleBuffer = [];
                ctx.postMessage({ 
                    type: 'FAULT_INJECTION_UPDATED',
                    payload: { faultInfo: payload.faultInfo }
                });
            }
            break;

        default:
            console.warn('Unknown message type:', type);
            break;
    }
});

// Add error handler for uncaught errors
ctx.addEventListener('error', function(error) {
    console.error('Worker global error:', error);
    ctx.postMessage({
        type: 'ERROR',
        payload: {
            message: 'Worker global error: ' + error.message,
            stack: error.stack
        }
    });
});

// Add unhandled rejection handler
ctx.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    ctx.postMessage({
        type: 'ERROR',
        payload: {
            message: 'Unhandled promise rejection: ' + event.reason,
            stack: event.reason.stack
        }
    });
});

export {};