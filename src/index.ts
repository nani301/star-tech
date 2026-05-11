/**
 * Face Liveness Detection - Main Entry Point
 * 
 * Government-grade browser-side face liveness verification.
 * 
 * Quick Start:
 * ```typescript
 * import {
 *   FaceLivenessVerifier,
 *   loadFaceModels,
 *   createFaceApiAdapter,
 *   enrollFace,
 * } from './face-liveness';
 * 
 * // 1. Load models
 * await loadFaceModels({ modelUri: '/models' });
 * 
 * // 2. Create adapter
 * const models = createFaceApiAdapter({ minConfidence: 0.7 });
 * 
 * // 3. Initialize verifier
 * const verifier = new FaceLivenessVerifier();
 * await verifier.initFaceVerifier(videoEl, canvasEl, models, {
 *   onStatus: (status, prompt) => console.log(status, prompt),
 *   onProgress: (progress, msg) => console.log(`${progress}%: ${msg}`),
 * });
 * 
 * // 4. Enroll user (one-time)
 * const enrolledDescriptor = await enrollFace(videoEl, 3);
 * 
 * // 5. Verify
 * const result = await verifier.startVerification(enrolledDescriptor);
 * if (result.success) {
 *   console.log('✅ Live face confirmed + identity matched');
 * } else {
 *   console.log('❌', result.reason, result.livenessScore);
 * }
 * ```
 */

// Core verifier
export { FaceLivenessVerifier, ModelInference } from './face-liveness-verifier';

// Face-api.js adapter
export {
  loadFaceModels,
  createFaceApiAdapter,
  enrollFace,
  ModelLoadOptions,
} from './face-api-adapter';

// Types
export {
  LivenessConfig,
  VerificationStatus,
  RejectionReason,
  UserPrompt,
  FaceDescriptor,
  VerificationResult,
  LivenessScore,
  AuditEntry,
  StatusCallback,
  ProgressCallback,
  ResultCallback,
  FrameRecord,
  BlinkEvent,
  HeadPose,
  FaceDetection,
  FaceLandmarks68,
  LANDMARK_INDICES,
} from './types';

// Config presets
export {
  DEFAULT_CONFIG,
  DEV_CONFIG,
  ULTRA_SECURE_CONFIG,
} from './config';

// Utilities (for advanced use cases)
export {
  calculateEAR,
  calculateMAR,
  estimateHeadPose,
  getNosePosition,
  getLandmarkBoundingBox,
  landmarkDifference,
  landmarkPositionVariance,
  computeFaceSimilarity,
  averageDescriptors,
} from './landmark-utils';

export {
  analyzeFrame,
  validateEnvironment,
  resetFrozenFrameDetector,
  computeFrameDifference,
  computeLaplacianVariance,
  detectMoirePattern,
  analyzeBrightness,
} from './frame-analyzer';

export {
  processBlinkFrame,
  analyzeHeadMovement,
  analyzeMicroMotion,
  analyzeTemporalConsistency,
  analyzeTexture,
  computeLivenessScore,
  makeLivenessDecision,
} from './liveness-engine';
