import { getCameraErrorMessage, logError } from '../utils/common.js';

export class CameraService {
  constructor() {
    this.stream = null;
    this.video = null;
    this.canvas = null;
    this.config = null;
    this.cameras = [];
    this.fps = 30;
    this.frameInterval = 1000 / 30;
    this.lastFrameTime = 0;
    this._cameraType = 'default';
  }

  setVideoElement(videoElement) {
    this.video = videoElement;
  }

  setCanvasElement(canvasElement) {
    this.canvas = canvasElement;
  }

  async loadCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.cameras = devices.filter((d) => d.kind === 'videoinput');
      return this.cameras;
    } catch (error) {
      logError('CameraService.loadCameras', error);
      return [];
    }
  }

  _getConstraints() {
    const facingMode = this._cameraType === 'front' ? 'user' : 'environment';
    return {
      video: {
        facingMode,
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };
  }

  async startCamera(cameraType = null) {
    if (cameraType !== null) {
      this._cameraType = cameraType;
    }

    try {
      if (this.stream) {
        this.stopCamera();
      }

      const constraints = this._getConstraints();
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (this.video) {
        this.video.srcObject = this.stream;
        await new Promise((resolve, reject) => {
          this.video.onloadedmetadata = resolve;
          this.video.onerror = reject;
          setTimeout(resolve, 3000);
        });
        await this.video.play();
      }

      return true;
    } catch (error) {
      logError('CameraService.startCamera', error);
      throw new Error(getCameraErrorMessage(error));
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  setFPS(fps) {
    this.fps = fps;
    this.frameInterval = 1000 / fps;
  }

  isThrottled(now) {
    if (now - this.lastFrameTime < this.frameInterval) return true;
    this.lastFrameTime = now;
    return false;
  }

  isActive() {
    return this.stream !== null && this.stream.active;
  }

  isReady() {
    return (
      this.video !== null &&
      this.video.readyState >= 2 &&
      this.video.videoWidth > 0
    );
  }
}
