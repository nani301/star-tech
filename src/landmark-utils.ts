/**
 * Landmark-Based Liveness Utilities
 * 
 * Pure functions for computing physiological signals from 68-point landmarks.
 * All functions are deterministic and side-effect free.
 * 
 * SECURITY: These calculations are the foundation of anti-spoof detection.
 * Any manipulation of landmark data would bypass liveness checks.
 * The caller must ensure landmarks come from the actual frame being analyzed.
 */

import { FaceLandmarks68, HeadPose, LANDMARK_INDICES } from './types';

// ─────────────────────────────────────────────────────────────
// Vector Math Helpers
// ─────────────────────────────────────────────────────────────

/** Euclidean distance between two points */
function dist(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

/** Get (x,y) from flattened landmark array */
function getPoint(landmarks: FaceLandmarks68, index: number): [number, number] {
  const arr = landmarks.positions;
  return [arr[index * 2], arr[index * 2 + 1]];
}

// ─────────────────────────────────────────────────────────────
// Eye Aspect Ratio (EAR)
// 
// EAR is a normalized measure of eye openness.
// It is invariant to scale and in-plane rotation.
// 
// Reference: "Real-Time Eye Blink Detection using Facial Landmarks"
// Soukupová and Čech, 2016
// 
// SECURITY: EAR is our primary signal for blink detection.
// A photo has constant EAR (no blink).
// A video replay has periodic EAR changes (detectable via timing).
// ─────────────────────────────────────────────────────────────

/**
 * Compute EAR for a single eye given 6 landmark indices.
 * 
 * Eye landmarks (clockwise from outer corner):
 *   p0 = outer corner, p1 = upper outer, p2 = upper inner,
 *   p3 = inner corner, p4 = lower inner, p5 = lower outer
 * 
 * EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
 */
function computeEyeEAR(
  landmarks: FaceLandmarks68,
  p0: number, p1: number, p2: number,
  p3: number, p4: number, p5: number
): number {
  const [x0, y0] = getPoint(landmarks, p0);
  const [x1, y1] = getPoint(landmarks, p1);
  const [x2, y2] = getPoint(landmarks, p2);
  const [x3, y3] = getPoint(landmarks, p3);
  const [x4, y4] = getPoint(landmarks, p4);
  const [x5, y5] = getPoint(landmarks, p5);

  const vertical1 = dist(x1, y1, x5, y5);
  const vertical2 = dist(x2, y2, x4, y4);
  const horizontal = dist(x0, y0, x3, y3);

  // Avoid division by zero (shouldn't happen with valid landmarks)
  if (horizontal === 0) return 1.0;

  return (vertical1 + vertical2) / (2.0 * horizontal);
}

/**
 * Compute EAR for both eyes and return average.
 * 
 * SECURITY NOTE: We average both eyes. Asymmetric blinks
 * (one eye closes, other doesn't) can indicate deepfake artifacts
 * or partial photo occlusion. We track symmetry separately.
 */
export function calculateEAR(landmarks: FaceLandmarks68): {
  average: number;
  left: number;
  right: number;
  symmetric: boolean;
} {
  // Right eye: indices 36-41
  const rightEAR = computeEyeEAR(landmarks, 36, 37, 38, 39, 40, 41);
  // Left eye: indices 42-47
  const leftEAR = computeEyeEAR(landmarks, 42, 43, 44, 45, 46, 47);

  const average = (leftEAR + rightEAR) / 2.0;

  // Symmetric if both eyes have similar EAR (within 30% of each other)
  const maxEAR = Math.max(leftEAR, rightEAR);
  const minEAR = Math.min(leftEAR, rightEAR);
  const symmetric = maxEAR === 0 ? true : (minEAR / maxEAR) > 0.7;

  return { average, left: leftEAR, right: rightEAR, symmetric };
}

// ─────────────────────────────────────────────────────────────
// Mouth Aspect Ratio (MAR)
// 
// Used for detecting mouth movements (breathing, speaking micro-motions)
// which are present in live humans but absent in photos.
// ─────────────────────────────────────────────────────────────

/**
 * Compute Mouth Aspect Ratio.
 * 
 * Outer mouth: 48 (left corner) → 54 (right corner)
 * Upper lip: 51, 52, 53
 * Lower lip: 57, 58, 59
 */
export function calculateMAR(landmarks: FaceLandmarks68): number {
  const [x48, y48] = getPoint(landmarks, 48);
  const [x54, y54] = getPoint(landmarks, 54);
  const [x51, y51] = getPoint(landmarks, 51);
  const [x57, y57] = getPoint(landmarks, 57);

  const mouthWidth = dist(x48, y48, x54, y54);
  const mouthHeight = dist(x51, y51, x57, y57);

  if (mouthWidth === 0) return 0;
  return mouthHeight / mouthWidth;
}

// ─────────────────────────────────────────────────────────────
// Head Pose Estimation (Approximate from Landmarks)
// 
// We estimate head pose using geometric relationships between
// facial landmarks. This is approximate but sufficient for
// liveness detection (we only need relative changes, not absolute angles).
// 
// SECURITY: Head pose changes over time prove 3D structure.
// A flat photo cannot produce natural head pose changes.
// ─────────────────────────────────────────────────────────────

/**
 * Estimate head pose from 68-point landmarks.
 * 
 * Uses simple geometric heuristics:
 * - Yaw:   based on nose position relative to eye centers
 * - Pitch: based on nose tip vertical position relative to eyes
 * - Roll:  based on angle between eye centers
 * 
 * These are APPROXIMATE angles in degrees, useful for relative
 * change detection, not absolute measurement.
 */
export function estimateHeadPose(landmarks: FaceLandmarks68): HeadPose {
  const [x30, y30] = getPoint(landmarks, 30); // Nose tip
  const [x27, y27] = getPoint(landmarks, 27); // Top of nose bridge
  const [x8, y8] = getPoint(landmarks, 8);   // Chin

  // Eye centers
  const [x36, y36] = getPoint(landmarks, 36);
  const [x39, y39] = getPoint(landmarks, 39);
  const [x42, y42] = getPoint(landmarks, 42);
  const [x45, y45] = getPoint(landmarks, 45);

  const rightEyeCenterX = (x36 + x39) / 2;
  const rightEyeCenterY = (y36 + y39) / 2;
  const leftEyeCenterX = (x42 + x45) / 2;
  const leftEyeCenterY = (y42 + y45) / 2;

  const eyeCenterX = (rightEyeCenterX + leftEyeCenterX) / 2;
  const eyeCenterY = (rightEyeCenterY + leftEyeCenterY) / 2;

  // Face width (distance between outer eye corners)
  const faceWidth = dist(x36, y36, x45, y45);
  // Face height (eyes to chin)
  const faceHeight = dist(eyeCenterX, eyeCenterY, x8, y8);

  // Avoid division by zero
  const safeWidth = faceWidth || 1;
  const safeHeight = faceHeight || 1;

  // Yaw: nose tip offset from face center, normalized by face width
  // Positive = facing left, Negative = facing right
  const yaw = ((x30 - eyeCenterX) / safeWidth) * 90;

  // Pitch: nose tip vertical offset, normalized by face height
  // Positive = looking up, Negative = looking down
  const pitch = ((eyeCenterY - y30) / safeHeight) * 90;

  // Roll: angle between eye centers
  const rollRad = Math.atan2(leftEyeCenterY - rightEyeCenterY, leftEyeCenterX - rightEyeCenterX);
  const roll = rollRad * (180 / Math.PI);

  return { pitch, yaw, roll };
}

// ─────────────────────────────────────────────────────────────
// Nose Position (for head tracking)
// ─────────────────────────────────────────────────────────────

/**
 * Get nose tip position for head movement tracking.
 * Nose tip (landmark 30) is the most stable central point.
 */
export function getNosePosition(landmarks: FaceLandmarks68): { x: number; y: number } {
  return {
    x: landmarks.positions[30 * 2],
    y: landmarks.positions[30 * 2 + 1],
  };
}

// ─────────────────────────────────────────────────────────────
// Face Bounding Box from Landmarks
// ─────────────────────────────────────────────────────────────

/**
 * Compute tight bounding box from landmarks.
 * Used to validate face size and detect sudden jumps.
 */
export function getLandmarkBoundingBox(landmarks: FaceLandmarks68): {
  x: number; y: number; width: number; height: number;
} {
  const arr = landmarks.positions;
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;

  for (let i = 0; i < 68; i++) {
    const x = arr[i * 2];
    const y = arr[i * 2 + 1];
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

// ─────────────────────────────────────────────────────────────
// Landmark Stability Check
// 
// SECURITY: Detects frozen frames, video loops, and injection attacks.
// Real human landmarks have continuous micro-variation.
// ─────────────────────────────────────────────────────────────

/**
 * Compute mean absolute difference between two landmark sets.
 * Returns average pixel displacement per landmark.
 */
export function landmarkDifference(
  a: FaceLandmarks68,
  b: FaceLandmarks68
): number {
  const arrA = a.positions;
  const arrB = b.positions;
  let sum = 0;

  for (let i = 0; i < arrA.length; i++) {
    sum += Math.abs(arrA[i] - arrB[i]);
  }

  return sum / arrA.length;
}

/**
 * Compute variance of a landmark position over time.
 * Low variance = static image or frozen frame.
 */
export function landmarkPositionVariance(
  positions: Array<{ x: number; y: number }>
): number {
  if (positions.length < 2) return 0;

  let sumX = 0, sumY = 0;
  for (const p of positions) {
    sumX += p.x;
    sumY += p.y;
  }
  const meanX = sumX / positions.length;
  const meanY = sumY / positions.length;

  let varX = 0, varY = 0;
  for (const p of positions) {
    varX += (p.x - meanX) ** 2;
    varY += (p.y - meanY) ** 2;
  }

  return (varX + varY) / positions.length;
}

// ─────────────────────────────────────────────────────────────
// Face Descriptor Comparison
// 
// Cosine similarity between two 128-dim face descriptors.
// 
// SECURITY: Descriptors must be extracted from the SAME frame
// that passed liveness checks. Do NOT use a cached descriptor
// from a previous session.
// ─────────────────────────────────────────────────────────────

/**
 * Compute cosine similarity between two face descriptors.
 * Range: [-1, 1], where 1 = identical, 0 = orthogonal.
 * 
 * The recognition model outputs L2-normalized descriptors,
 * so cosine similarity = dot product.
 */
export function computeFaceSimilarity(
  a: Float32Array,
  b: Float32Array
): number {
  if (a.length !== 128 || b.length !== 128) {
    throw new Error('Face descriptors must be 128-dimensional');
  }

  let dot = 0;
  for (let i = 0; i < 128; i++) {
    dot += a[i] * b[i];
  }

  // Clamp to valid range (protect against floating point drift)
  return Math.max(-1, Math.min(1, dot));
}

/**
 * Average multiple face descriptors.
 * Used for enrollment to reduce noise.
 */
export function averageDescriptors(descriptors: Float32Array[]): Float32Array {
  if (descriptors.length === 0) {
    throw new Error('Cannot average empty descriptor array');
  }

  const result = new Float32Array(128);
  for (const desc of descriptors) {
    for (let i = 0; i < 128; i++) {
      result[i] += desc[i];
    }
  }

  for (let i = 0; i < 128; i++) {
    result[i] /= descriptors.length;
  }

  // Re-normalize (L2)
  let norm = 0;
  for (let i = 0; i < 128; i++) {
    norm += result[i] ** 2;
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let i = 0; i < 128; i++) {
      result[i] /= norm;
    }
  }

  return result;
}
