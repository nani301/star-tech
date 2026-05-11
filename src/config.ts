/**
 * Default Liveness Configuration
 * 
 * SECURITY NOTE: These defaults are calibrated for high-security
 * scenarios (government, banking, critical infrastructure).
 * They prioritize security over user convenience.
 * 
 * For lower-security use cases (e.g., casual attendance),
 * relax: earThreshold, minBlinksRequired, observationWindowMs,
 *        livenessPassThreshold, faceMatchThreshold
 */

import { LivenessConfig } from './types';

export const DEFAULT_CONFIG: LivenessConfig = {
  // ── Face Detection ──
  // High threshold reduces false positives from background faces
  faceDetectionThreshold: 0.7,
  // Face must be at least 30% of frame width for reliable landmarks
  minFaceSizeRatio: 0.25,
  // Face shouldn't fill entire frame (could be too close / photo)
  maxFaceSizeRatio: 0.85,

  // ── Temporal Observation ──
  // Minimum 2.5 seconds of observation (prevents single-frame attacks)
  observationWindowMs: 2500,
  // At 30fps, need at least 45 frames
  minFramesForDecision: 45,
  // Circular buffer: 4 seconds at 30fps
  frameBufferSize: 120,
  // Target processing rate
  targetFps: 30,

  // ── Blink Detection ──
  // EAR < 0.2 means eye is significantly closed
  // Calibrated on dlib 68-point model
  earThreshold: 0.2,
  // Eye must be closed for at least 3 frames (~100ms at 30fps)
  // Prevents "instant blink" from video cuts
  blinkMinClosedFrames: 3,
  // Eye closed for more than 15 frames (~500ms) is suspicious
  // Could be a photo with eyes closed, or unnatural hold
  blinkMaxClosedFrames: 15,
  // Eye must reopen and stay open for 2 frames
  blinkMinOpenFrames: 2,
  // Require at least 1 natural blink during observation
  minBlinksRequired: 1,
  // Natural human blink: 100-400ms
  maxBlinkDurationMs: 500,
  minBlinkDurationMs: 80,

  // ── Head Movement ──
  // Minimum 15 pixels of movement (relative to detection resolution)
  minHeadMovementPixels: 15,
  // Reject jumps > 30 pixels between frames (injection attack)
  maxHeadJumpPixels: 30,
  // Minimum variance in position over tracking window
  minHeadVariance: 5,
  // Track last 30 frames for movement analysis
  headTrackingWindow: 30,

  // ── Frame Stability / Anti-Spoof ──
  // Frame difference < 2.0 is essentially identical
  frozenFramePixelThreshold: 2.0,
  // Allow at most 5 consecutive frozen frames
  // (brief freezes can happen from browser GC)
  maxFrozenFrames: 5,
  // Landmark difference < 0.5 is essentially identical
  frozenFrameLandmarkThreshold: 0.5,
  // Moiré sensitivity (0-1, higher = more sensitive)
  moireDetectionSensitivity: 0.3,

  // ── Brightness / Environment ──
  // Reject very dark scenes (< 15% brightness)
  minBrightness: 40,
  // Reject overexposed scenes (> 95% brightness)
  maxBrightness: 240,
  // Reject perfectly uniform images (photos, screens)
  minBrightnessVariance: 15,

  // ── Identity Matching ──
  // Cosine similarity threshold for face match
  // 0.5 = high confidence match (face-api.js default)
  faceMatchThreshold: 0.5,
  // Average 3 samples for enrollment descriptor
  enrollmentSamples: 3,

  // ── Timing & Retries ──
  // Total timeout: 15 seconds
  verificationTimeoutMs: 15000,
  // Allow 2 retries before final rejection
  maxRetries: 2,
  // 1 second delay between retries
  retryDelayMs: 1000,

  // ── Liveness Fusion ──
  // Must score >= 0.8 to pass (high bar)
  livenessPassThreshold: 0.8,
  // Score < 0.5 is immediate fail
  livenessFailThreshold: 0.5,
  // Weights for multi-signal fusion
  // Blink is strongest signal for anti-photo
  weightBlink: 0.30,
  // Head movement catches replay and deepfake
  weightHeadMovement: 0.25,
  // Micro-motion catches static images
  weightMicroMotion: 0.20,
  // Temporal consistency catches frozen frames
  weightTemporal: 0.15,
  // Texture catches screen replay
  weightTexture: 0.10,
};

/**
 * Relaxed configuration for development/testing.
 * NOT for production use.
 */
export const DEV_CONFIG: LivenessConfig = {
  ...DEFAULT_CONFIG,
  faceDetectionThreshold: 0.5,
  observationWindowMs: 1500,
  minFramesForDecision: 20,
  minBlinksRequired: 1,
  earThreshold: 0.22,
  livenessPassThreshold: 0.6,
  verificationTimeoutMs: 10000,
  maxRetries: 3,
};

/**
 * Ultra-secure configuration for maximum security scenarios.
 * May have higher false rejection rate.
 */
export const ULTRA_SECURE_CONFIG: LivenessConfig = {
  ...DEFAULT_CONFIG,
  faceDetectionThreshold: 0.8,
  observationWindowMs: 4000,
  minFramesForDecision: 90,
  minBlinksRequired: 2,
  earThreshold: 0.18,
  livenessPassThreshold: 0.9,
  minHeadMovementPixels: 25,
  maxFrozenFrames: 2,
  verificationTimeoutMs: 20000,
  maxRetries: 1,
  weightBlink: 0.35,
  weightHeadMovement: 0.30,
  weightMicroMotion: 0.20,
  weightTemporal: 0.10,
  weightTexture: 0.05,
};
