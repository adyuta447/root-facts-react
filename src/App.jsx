import { useRef, useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import CameraSection from './components/CameraSection';
import InfoPanel from './components/InfoPanel';
import { useAppState } from './hooks/useAppState';
import { DetectionService } from './services/DetectionService';
import { CameraService } from './services/CameraService';
import { RootFactsService } from './services/RootFactsService';
import { isValidDetection, APP_CONFIG } from './utils/config';
import { createDelay } from './utils/common';

function App() {
  const { state, actions } = useAppState();
  const detectionCleanupRef = useRef(null);
  const isRunningRef = useRef(false);
  const [currentTone, setCurrentTone] = useState('normal');

  const detectorRef = useRef(null);
  const cameraRef = useRef(null);
  const generatorRef = useRef(null);
  const lastPredictionTimeRef = useRef(0);
  const stableCountRef = useRef(0);
  const lastDetectedClassRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    const initServices = async () => {
      const detector = new DetectionService();
      const camera = new CameraService();
      const generator = new RootFactsService();

      detectorRef.current = detector;
      cameraRef.current = camera;
      generatorRef.current = generator;

      actions.setServices({ detector, camera, generator });
      actions.setModelStatus('Memuat Model... 0%');

      try {
        await detector.loadModel((progress) => {
          if (!cancelled) {
            actions.setModelStatus(`Memuat Model... ${progress}%`);
          }
        });

        if (cancelled) return;

        if (!detector.isLoaded()) {
          actions.setModelStatus('Gagal Memuat Model');
          return;
        }

        if (!cancelled) actions.setModelStatus('Memuat Model AI...');

        try {
          await generator.loadModel((progress) => {
            if (!cancelled) actions.setModelStatus(`Memuat Model AI... ${progress}%`);
          });
          if (!cancelled) actions.setModelStatus('Model AI Siap');
        } catch (err) {
          console.warn('LLM gagal dimuat, fun fact tidak tersedia:', err);
          if (!cancelled) actions.setModelStatus('Model Deteksi Siap');
        }
      } catch (error) {
        if (!cancelled) {
          actions.setError(`Gagal memuat model deteksi: ${  error.message}`);
          actions.setModelStatus('Gagal Memuat Model');
        }
      }
    };

    initServices();

    return () => {
      cancelled = true;
      isRunningRef.current = false;
      if (detectionCleanupRef.current) {
        detectionCleanupRef.current();
      }
      cameraRef.current?.stopCamera();
    };
  }, []);

  const startDetectionLoop = useCallback(() => {
    isRunningRef.current = true;
    lastPredictionTimeRef.current = 0;
    stableCountRef.current = 0;
    lastDetectedClassRef.current = null;

    const loop = async () => {
      if (!isRunningRef.current) return;

      const camera = cameraRef.current;
      const detector = detectorRef.current;

      const now = performance.now();
      if (camera.isThrottled(now)) {
        requestAnimationFrame(loop);
        return;
      }

      const timeSinceLastPrediction = now - lastPredictionTimeRef.current;
      if (timeSinceLastPrediction < APP_CONFIG.predictionInterval) {
        requestAnimationFrame(loop);
        return;
      }
      lastPredictionTimeRef.current = now;

      if (camera.isReady() && detector.isLoaded()) {
        try {
          const result = await detector.predict(camera.video);

          if (!isRunningRef.current) return;

          if (isValidDetection(result)) {
            if (result.className === lastDetectedClassRef.current) {
              stableCountRef.current += 1;
            } else {
              stableCountRef.current = 1;
              lastDetectedClassRef.current = result.className;
            }

            if (stableCountRef.current < APP_CONFIG.stableDetectionRequired) {
              requestAnimationFrame(loop);
              return;
            }

            isRunningRef.current = false;
            stableCountRef.current = 0;
            actions.setRunning(false);
            actions.setAppState('analyzing');

            await createDelay(APP_CONFIG.analyzingDelay);

            actions.setDetectionResult(result);
            actions.setAppState('result');
            actions.setFunFactData(null);

            camera.stopCamera();

            const generator = generatorRef.current;
            try {
              const fact = await generator.generateFacts(result.className);
              actions.setFunFactData(fact || 'error');
            } catch {
              actions.setFunFactData('error');
            }

            return;
          } else {
            stableCountRef.current = 0;
            lastDetectedClassRef.current = null;
          }
        } catch (err) {
          console.error('Prediction error:', err);
        }
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);

    detectionCleanupRef.current = () => {
      isRunningRef.current = false;
    };
  }, [actions]);

  const handleToggleCamera = useCallback(async () => {
    if (state.isRunning) {
      isRunningRef.current = false;
      actions.setRunning(false);
      actions.resetResults();
      cameraRef.current?.stopCamera();
    } else {
      actions.resetResults();
      actions.setRunning(true);

      try {
        await cameraRef.current?.startCamera();
        startDetectionLoop();
      } catch (error) {
        actions.setError(error.message);
        actions.setRunning(false);
      }
    }
  }, [state.isRunning, actions, startDetectionLoop]);

  const handleToneChange = useCallback((tone) => {
    setCurrentTone(tone);
    generatorRef.current?.setTone(tone);
  }, []);

  const handleCopyFact = useCallback(async () => {
    if (state.funFactData && state.funFactData !== 'error') {
      try {
        await navigator.clipboard.writeText(state.funFactData);
      } catch (err) {
        console.error('Gagal menyalin teks:', err);
      }
    }
  }, [state.funFactData]);

  return (
    <div className="app-container">
      <Header modelStatus={state.modelStatus} />

      <main className="main-content">
        <CameraSection
          isRunning={state.isRunning}
          onToggleCamera={handleToggleCamera}
          onToneChange={handleToneChange}
          services={state.services}
          modelStatus={state.modelStatus}
          error={state.error}
          currentTone={currentTone}
        />

        <InfoPanel
          appState={state.appState}
          detectionResult={state.detectionResult}
          funFactData={state.funFactData}
          error={state.error}
          onCopyFact={handleCopyFact}
        />
      </main>

      <footer className="footer">
        <p>Powered by TensorFlow.js & Transformers.js</p>
      </footer>

      {state.error && (
        <div style={{
          position: 'fixed',
          bottom: '1rem',
          left: '50%',
          transform: 'translateX(-50%)',
          maxWidth: '380px',
          padding: '0.875rem 1rem',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 'var(--radius-md)',
          color: '#991b1b',
          fontSize: '0.8125rem',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          zIndex: 1000
        }}>
          <strong>Error:</strong> {state.error}
          <button
            onClick={() => actions.setError(null)}
            style={{
              marginLeft: 'auto',
              background: 'transparent',
              border: 'none',
              fontSize: '1.25rem',
              cursor: 'pointer',
              color: '#991b1b',
              padding: 0,
              lineHeight: 1
            }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
