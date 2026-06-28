import * as tf from '@tensorflow/tfjs';
import { isWebGPUSupported, logError, validateModelMetadata } from '../utils/common.js';

const MODEL_URL = '/model/model.json';
const METADATA_URL = '/model/metadata.json';
const IMAGE_SIZE = 224;

export class DetectionService {
  constructor() {
    this.model = null;
    this.labels = [];
    this.config = null;
    this.currentBackend = null;
  }

  async loadModel(onProgress) {
    try {
      onProgress?.(5);

      if (isWebGPUSupported()) {
        try {
          await import('@tensorflow/tfjs-backend-webgpu');
          await tf.setBackend('webgpu');
          await tf.ready();
          this.currentBackend = 'webgpu';
        } catch {
          await tf.setBackend('webgl');
          await tf.ready();
          this.currentBackend = 'webgl';
        }
      } else {
        await tf.setBackend('webgl');
        await tf.ready();
        this.currentBackend = 'webgl';
      }

      onProgress?.(20);

      const metadataRes = await fetch(METADATA_URL);
      const metadata = await metadataRes.json();

      if (!validateModelMetadata(metadata)) {
        throw new Error('Invalid model metadata: missing labels');
      }

      onProgress?.(40);

      const model = await tf.loadLayersModel(MODEL_URL);

      onProgress?.(90);

      this.labels = metadata.labels;
      this.config = metadata;
      this.model = model;

      onProgress?.(100);
      return true;
    } catch (error) {
      logError('DetectionService.loadModel', error);
      throw error;
    }
  }

  async predict(imageElement) {
    if (!this.isLoaded()) return null;

    const scores = tf.tidy(() => {
      const img = tf.browser.fromPixels(imageElement);

      // Center crop to square — same as Teachable Machine's internal capture()
      const size = Math.min(img.shape[0], img.shape[1]);
      const beginH = Math.floor((img.shape[0] - size) / 2);
      const beginW = Math.floor((img.shape[1] - size) / 2);
      const cropped = img.slice([beginH, beginW, 0], [size, size, 3]);

      const input = cropped
        .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat()
        .sub(127.5)
        .div(127.5)
        .expandDims(0);

      const output = this.model.predict(input);
      return Array.from(output.dataSync());
    });

    let maxScore = 0;
    let maxIndex = 0;
    for (let i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIndex = i;
      }
    }

    return {
      className: this.labels[maxIndex],
      score: maxScore,
      confidence: Math.round(maxScore * 100),
      isValid: true,
    };
  }

  isLoaded() {
    return this.model !== null && this.labels.length > 0;
  }
}
