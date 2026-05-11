/**
 * Face-API.js Adapter
 * 
 * Bridges the FaceLivenessVerifier to the existing face-api.js model stack.
 * This adapter wraps face-api.js inference calls to match the ModelInference
 * interface expected by the verifier.
 * 
 * The existing models in this repo:
 * - SSD MobileNet v1 (face detection)
 * - Face Landmark 68 (68-point landmarks)
 * - Face Recognition (128-dim descriptors)
 * 
 * SECURITY: This adapter ensures that:
 * 1. Models are loaded from local paths (no external API calls)
 * 2. Inference runs entirely in the browser
 * 3. No data leaves the client
 */

import { ModelInference } from './face-liveness-verifier';

// ─────────────────────────────────────────────────────────────
// Type declarations for face-api.js (to avoid direct import issues)
// These match the face-api.js API surface we need.
// ─────────────────────────────────────────────────────────────

declare global {
  interface Window {
    faceapi: {
      nets: {
        ssdMobilenetv1: FaceApiNet;
        faceLandmark68Net: FaceApiNet;
        faceRecognitionNet: FaceApiNet;
      };
      detectAllFaces: (input: HTMLVideoElement | HTMLCanvasElement) => {
        withFaceLandmarks: () => Promise<FaceApiDetection[]>;
      };
      detectSingleFace: (input: HTMLVideoElement | HTMLCanvasElement) => {
        withFaceLandmarks: () => {
          withFaceDescriptor: () => Promise<FaceApiDetection | undefined>;
        };
      };
      SsdMobilenetv1Options: new (opts?: { minConfidence?: number; maxResults?: number }) => unknown;
      euclideanDistance: (a: Float32Array, b: Float32Array) => number;
    };
  }
}

interface FaceApiNet {
  loadFromUri(uri: string): Promise<void>;
  load(weights: unknown): Promise<void>;
}

interface FaceApiDetection {
  detection: {
    box: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    score: number;
    classScore: number;
  };
  landmarks?: {
    positions: Float32Array;
    getJawOutline(): Array<{ x: number; y: number }>;
    getLeftEye(): Array<{ x: number; y: number }>;
    getRightEye(): Array<{ x: number; y: number }>;
  };
  descriptor?: Float32Array;
  alignedRect?: {
    box: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
  };
}

// ─────────────────────────────────────────────────────────────
// Model Loading
// ─────────────────────────────────────────────────────────────

export interface ModelLoadOptions {
  /** Base path to model weights (e.g., '/models' or './weights') */
  modelUri: string;
  /** Detection confidence threshold */
  minConfidence?: number;
  /** Maximum concurrent model loads */
  maxResults?: number;
}

/**
 * Load all three face models from local paths.
 * 
 * SECURITY: Models must be loaded from local server paths.
 * Do NOT load from external CDNs or APIs.
 */
export async function loadFaceModels(options: ModelLoadOptions): Promise<void> {
  const { modelUri, minConfidence = 0.5, maxResults = 1 } = options;

  if (!window.faceapi) {
    throw new Error(
      'face-api.js not loaded. Ensure the library is included before loading models.'
    );
  }

  const loadPromises = [
    window.faceapi.nets.ssdMobilenetv1.loadFromUri(modelUri),
    window.faceapi.nets.faceLandmark68Net.loadFromUri(modelUri),
    window.faceapi.nets.faceRecognitionNet.loadFromUri(modelUri),
  ];

  await Promise.all(loadPromises);
}

// ─────────────────────────────────────────────────────────────
// Model Inference Adapter
// 
// Wraps face-api.js to implement the ModelInference interface.
// ─────────────────────────────────────────────────────────────

/**
 * Create a ModelInference adapter from loaded face-api.js models.
 * 
 * This adapter is passed to FaceLivenessVerifier.initFaceVerifier().
 */
export function createFaceApiAdapter(
  options: { minConfidence?: number; maxResults?: number } = {}
): ModelInference {
  const { minConfidence = 0.5, maxResults = 1 } = options;

  return {
    /**
     * Detect faces using SSD MobileNet v1.
     * 
     * Returns detections with bounding boxes and confidence scores.
     * Only returns the highest-confidence detection (maxResults=1)
     * for liveness verification (multi-face = security risk).
     */
    async detectFaces(input: HTMLVideoElement | HTMLCanvasElement) {
      if (!window.faceapi) {
        throw new Error('face-api.js not available');
      }

      const ssdOptions = new window.faceapi.SsdMobilenetv1Options({
        minConfidence,
        maxResults,
      });

      const detections = await window.faceapi
        .detectAllFaces(input, ssdOptions)
        .withFaceLandmarks();

      return detections.map(d => ({
        box: {
          x: d.detection.box.x,
          y: d.detection.box.y,
          width: d.detection.box.width,
          height: d.detection.box.height,
        },
        score: d.detection.score,
      }));
    },

    /**
     * Detect 68-point landmarks for a given face box.
     * 
     * Uses the faceLandmark68Net model.
     * Returns 136 values: 68 (x, y) pairs.
     */
    async detectLandmarks(
      input: HTMLVideoElement | HTMLCanvasElement,
      box: { x: number; y: number; width: number; height: number }
    ) {
      if (!window.faceapi) {
        throw new Error('face-api.js not available');
      }

      // face-api.js detectSingleFace returns landmarks aligned to the face
      const result = await window.faceapi
        .detectSingleFace(input)
        .withFaceLandmarks();

      if (!result || !result.landmarks) {
        return null;
      }

      // Extract 68 (x,y) positions as flat Float32Array
      const positions = result.landmarks.positions;
      const flatArray = new Float32Array(136);
      for (let i = 0; i < 68; i++) {
        flatArray[i * 2] = positions[i].x;
        flatArray[i * 2 + 1] = positions[i].y;
      }

      return flatArray;
    },

    /**
     * Compute 128-dim face descriptor for identity matching.
     * 
     * Uses the faceRecognitionNet model.
     * Descriptor is L2-normalized by the model.
     */
    async computeDescriptor(
      input: HTMLVideoElement | HTMLCanvasElement,
      box: { x: number; y: number; width: number; height: number }
    ) {
      if (!window.faceapi) {
        throw new Error('face-api.js not available');
      }

      const result = await window.faceapi
        .detectSingleFace(input)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!result || !result.descriptor) {
        return null;
      }

      return result.descriptor;
    },
  };
}

// ─────────────────────────────────────────────────────────────
// Enrollment Helper
// 
 * Convenience function for enrolling a new user.
 * Captures multiple descriptors and averages them.
// ─────────────────────────────────────────────────────────────

/**
 * Enroll a user by capturing their face descriptor.
 * 
 * @param videoEl - Active webcam video element
 * @param samples - Number of samples to average (default: 3)
 * @returns Averaged 128-dim face descriptor
 */
export async function enrollFace(
  videoEl: HTMLVideoElement,
  samples: number = 3,
  minConfidence: number = 0.7
): Promise<Float32Array | null> {
  if (!window.faceapi) {
    throw new Error('face-api.js not available');
  }

  const descriptors: Float32Array[] = [];
  const ssdOptions = new window.faceapi.SsdMobilenetv1Options({
    minConfidence,
    maxResults: 1,
  });

  for (let i = 0; i < samples; i++) {
    const result = await window.faceapi
      .detectSingleFace(videoEl, ssdOptions)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (result && result.descriptor) {
      descriptors.push(result.descriptor);
    }

    // Small delay between samples for natural variation
    if (i < samples - 1) {
      await new Promise(r => setTimeout(r, 300));
    }
  }

  if (descriptors.length === 0) {
    return null;
  }

  // Average and re-normalize
  const avg = new Float32Array(128);
  for (const desc of descriptors) {
    for (let i = 0; i < 128; i++) {
      avg[i] += desc[i];
    }
  }
  for (let i = 0; i < 128; i++) {
    avg[i] /= descriptors.length;
  }

  // Re-normalize to unit length
  let norm = 0;
  for (let i = 0; i < 128; i++) {
    norm += avg[i] * avg[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < 128; i++) {
      avg[i] /= norm;
    }
  }

  return avg;
}
