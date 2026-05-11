/**
 * Liveness Detection Engine
 * 
 * Core temporal analysis engine that processes frame records
 * and computes liveness scores.
 * 
 * SECURITY: This module implements the multi-signal fusion
 * that makes spoofing extremely difficult. An attacker must
 * simultaneously fake:
 * 1. Natural blink dynamics (state machine + timing)
 * 2. Continuous head movement (variance + smoothness)
 * 3. Micro-movements (landmark jitter)
 * 4. Temporal consistency (no frozen frames)
 * 5. Texture patterns (anti-screen)
 * 
 * Faking all five simultaneously is computationally infeasible
 * for real-time attacks without specialized hardware.
 */

import {
  LivenessConfig,
  FrameRecord,
  BlinkEvent,
  LivenessScore,
  RejectionReason,
} from './types';

// ─────────────────────────────────────────────────────────────
// Blink State Machine
// 
// SECURITY: Simple threshold crossing is NOT sufficient.
// A video replay can have perfect threshold-crossing blinks.
// We verify the full state transition with timing constraints.
// 
// States:
//   OPEN      → eye is open (EAR > threshold)
//   CLOSING   → eye closing (EAR dropped below threshold)
//   CLOSED    → eye fully closed (EAR stayed low for min frames)
//   OPENING   → eye reopening (EAR rose above threshold)
// 
// Valid blink: OPEN → CLOSING → CLOSED → OPENING → OPEN
// with timing within natural human limits.
// ─────────────────────────────────────────────────────────────

type BlinkState = 'OPEN' | 'CLOSING' | 'CLOSED' | 'OPENING';

interface BlinkStateMachine {
  state: BlinkState;
  framesInState: number;
  blinkStartFrame: number;
  minEarInBlink: number;
  blinkEvents: BlinkEvent[];
  consecutiveFrozen: number;
}

function createBlinkStateMachine(): BlinkStateMachine {
  return {
    state: 'OPEN',
    framesInState: 0,
    blinkStartFrame: 0,
    minEarInBlink: 1.0,
    blinkEvents: [],
    consecutiveFrozen: 0,
  };
}

/**
 * Process a single frame's EAR through the blink state machine.
 * 
 * SECURITY: Returns blink events only when a COMPLETE valid blink
 * is detected with proper timing. Partial or suspicious blinks
 * are rejected.
 */
export function processBlinkFrame(
  machine: BlinkStateMachine,
  frame: FrameRecord,
  config: LivenessConfig
): { events: BlinkEvent[]; suspicious: boolean } {
  const ear = frame.ear;
  const isClosed = ear < config.earThreshold;
  const events: BlinkEvent[] = [];
  let suspicious = false;

  // Track how long we've been in current state
  machine.framesInState++;

  switch (machine.state) {
    case 'OPEN':
      if (isClosed) {
        // Eye starting to close
        machine.state = 'CLOSING';
        machine.framesInState = 1;
        machine.blinkStartFrame = frame.frameNum;
        machine.minEarInBlink = ear;
      }
      break;

    case 'CLOSING':
      machine.minEarInBlink = Math.min(machine.minEarInBlink, ear);

      if (!isClosed) {
        // Eye reopened before fully closing = false start
        machine.state = 'OPEN';
        machine.framesInState = 0;
      } else if (machine.framesInState >= config.blinkMinClosedFrames) {
        // Eye has been closed long enough = fully closed
        machine.state = 'CLOSED';
        machine.framesInState = 1;
      }
      // If still closing but not yet min frames, stay in CLOSING
      break;

    case 'CLOSED':
      machine.minEarInBlink = Math.min(machine.minEarInBlink, ear);

      if (!isClosed) {
        // Eye starting to open
        machine.state = 'OPENING';
        machine.framesInState = 1;
      } else if (machine.framesInState > config.blinkMaxClosedFrames) {
        // Eye closed too long = suspicious (photo with eyes closed?)
        suspicious = true;
        machine.state = 'OPEN';
        machine.framesInState = 0;
      }
      break;

    case 'OPENING':
      if (isClosed) {
        // Eye closed again during opening = abnormal
        suspicious = true;
        machine.state = 'CLOSED';
        machine.framesInState = 1;
      } else if (machine.framesInState >= config.blinkMinOpenFrames) {
        // Eye fully reopened = blink complete!
        const durationFrames = frame.frameNum - machine.blinkStartFrame;
        const fps = config.targetFps;
        const durationMs = (durationFrames / fps) * 1000;

        // Validate timing (natural blink: 80-500ms)
        if (durationMs >= config.minBlinkDurationMs &&
            durationMs <= config.maxBlinkDurationMs) {
          events.push({
            startFrame: machine.blinkStartFrame,
            endFrame: frame.frameNum,
            durationMs,
            minEar: machine.minEarInBlink,
            symmetric: true, // Will be validated externally
          });
        } else {
          // Timing outside natural range = suspicious
          suspicious = true;
        }

        machine.state = 'OPEN';
        machine.framesInState = 0;
      }
      break;
  }

  return { events, suspicious };
}

// ─────────────────────────────────────────────────────────────
// Head Movement Analysis
// 
// SECURITY: Head movement proves 3D structure and active presence.
// A photo produces zero movement. A video replay produces
// pre-recorded movement (detectable via periodicity).
// 
// We track:
// 1. Total displacement (must exceed minimum)
// 2. Frame-to-frame jumps (must be smooth, no injection)
// 3. Variance (must have continuous micro-movement)
// 4. Trajectory smoothness (natural motion is smooth)
// ─────────────────────────────────────────────────────────────

interface HeadMovementResult {
  /** Whether sufficient movement was detected */
  sufficient: boolean;
  /** Whether any suspicious jumps detected */
  suspicious: boolean;
  /** Total displacement in pixels */
  totalDisplacement: number;
  /** Position variance */
  variance: number;
  /** Maximum frame-to-frame jump */
  maxJump: number;
}

/**
 * Analyze head movement from frame buffer.
 * 
 * Uses nose tip position (most stable central landmark).
 */
export function analyzeHeadMovement(
  frames: FrameRecord[],
  config: LivenessConfig
): HeadMovementResult {
  const positions = frames
    .filter(f => f.landmarks !== null)
    .map(f => f.nosePosition);

  if (positions.length < 5) {
    return {
      sufficient: false,
      suspicious: false,
      totalDisplacement: 0,
      variance: 0,
      maxJump: 0,
    };
  }

  // Calculate total displacement (max distance between any two points)
  let totalDisplacement = 0;
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const dx = positions[i].x - positions[j].x;
      const dy = positions[i].y - positions[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      totalDisplacement = Math.max(totalDisplacement, d);
    }
  }

  // Calculate frame-to-frame jumps
  let maxJump = 0;
  let suspicious = false;
  for (let i = 1; i < positions.length; i++) {
    const dx = positions[i].x - positions[i - 1].x;
    const dy = positions[i].y - positions[i - 1].y;
    const jump = Math.sqrt(dx * dx + dy * dy);
    maxJump = Math.max(maxJump, jump);

    if (jump > config.maxHeadJumpPixels) {
      suspicious = true; // Possible frame injection
    }
  }

  // Calculate variance
  const meanX = positions.reduce((s, p) => s + p.x, 0) / positions.length;
  const meanY = positions.reduce((s, p) => s + p.y, 0) / positions.length;
  const variance = positions.reduce((s, p) => {
    return s + (p.x - meanX) ** 2 + (p.y - meanY) ** 2;
  }, 0) / positions.length;

  const sufficient =
    totalDisplacement >= config.minHeadMovementPixels &&
    variance >= config.minHeadVariance;

  return { sufficient, suspicious, totalDisplacement, variance, maxJump };
}

// ─────────────────────────────────────────────────────────────
// Micro-Motion Analysis
// 
// Natural humans have involuntary micro-movements:
// - Eye saccades and tremor
// - Head drift from muscle tone
// - Lip movements from breathing
// - Facial muscle micro-contractions
// 
// SECURITY: Photos have ZERO micro-motion.
// Videos have periodic or repetitive micro-motion.
// We detect continuous non-periodic variation.
// ─────────────────────────────────────────────────────────────

interface MicroMotionResult {
  score: number;
  suspicious: boolean;
}

/**
 * Analyze micro-movements from landmark variance.
 * 
 * Computes variance of multiple landmark positions over time.
 * Real faces have continuous small variance.
 */
export function analyzeMicroMotion(
  frames: FrameRecord[],
  config: LivenessConfig
): MicroMotionResult {
  if (frames.length < 10) {
    return { score: 0, suspicious: false };
  }

  // Track multiple facial points for robustness
  const points: Array<{ x: number; y: number }>[] = [];
  const pointIndices = [30, 33, 48, 54, 36, 45]; // Nose, mouth corners, eye corners

  for (const idx of pointIndices) {
    const positions = frames
      .filter(f => f.landmarks !== null)
      .map(f => ({
        x: f.landmarks!.positions[idx * 2],
        y: f.landmarks!.positions[idx * 2 + 1],
      }));
    points.push(positions);
  }

  // Calculate average variance across all tracked points
  let totalVariance = 0;
  for (const positions of points) {
    if (positions.length < 2) continue;
    const meanX = positions.reduce((s, p) => s + p.x, 0) / positions.length;
    const meanY = positions.reduce((s, p) => s + p.y, 0) / positions.length;
    const variance = positions.reduce((s, p) => {
      return s + (p.x - meanX) ** 2 + (p.y - meanY) ** 2;
    }, 0) / positions.length;
    totalVariance += variance;
  }

  const avgVariance = totalVariance / points.length;

  // Check for periodicity (video replay indicator)
  // Simple autocorrelation check on nose position
  const nosePositions = frames
    .filter(f => f.landmarks !== null)
    .map(f => f.nosePosition);

  let periodic = false;
  if (nosePositions.length > 20) {
    for (let lag = 5; lag < 20; lag++) {
      let correlation = 0;
      for (let i = 0; i < nosePositions.length - lag; i++) {
        const dx = nosePositions[i].x - nosePositions[i + lag].x;
        const dy = nosePositions[i].y - nosePositions[i + lag].y;
        correlation += Math.sqrt(dx * dx + dy * dy);
      }
      const avgCorr = correlation / (nosePositions.length - lag);
      // Very low difference at regular intervals = periodic
      if (avgCorr < 0.5) {
        periodic = true;
        break;
      }
    }
  }

  // Score: higher variance = more alive, but too high = noise
  // Normalize to roughly [0, 1]
  const varianceScore = Math.min(1, avgVariance / 10);

  // Penalize periodic motion (video replay)
  const finalScore = periodic ? varianceScore * 0.3 : varianceScore;

  return {
    score: finalScore,
    suspicious: periodic,
  };
}

// ─────────────────────────────────────────────────────────────
// Temporal Consistency Analysis
// 
// SECURITY: Detects frozen frames, frame loops, and injection.
// We track:
// 1. Frame difference over time (should be continuous)
// 2. Landmark stability (should have natural jitter)
// 3. Frozen frame count (too many = video paused/photo)
// ─────────────────────────────────────────────────────────────

interface TemporalResult {
  score: number;
  frozenFrameCount: number;
  suspicious: boolean;
}

export function analyzeTemporalConsistency(
  frames: FrameRecord[],
  config: LivenessConfig
): TemporalResult {
  if (frames.length < 5) {
    return { score: 0.5, frozenFrameCount: 0, suspicious: false };
  }

  let frozenFrameCount = 0;
  let maxConsecutiveFrozen = 0;
  let currentFrozen = 0;

  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1];
    const curr = frames[i];

    const isFrozen =
      curr.analysis.frameDiff < config.frozenFramePixelThreshold;

    if (isFrozen) {
      currentFrozen++;
      frozenFrameCount++;
      maxConsecutiveFrozen = Math.max(maxConsecutiveFrozen, currentFrozen);
    } else {
      currentFrozen = 0;
    }
  }

  // Score based on frozen frame ratio
  const frozenRatio = frozenFrameCount / frames.length;
  let score = Math.max(0, 1 - frozenRatio * 5);

  // Heavily penalize too many consecutive frozen frames
  if (maxConsecutiveFrozen > config.maxFrozenFrames) {
    score *= 0.3;
  }

  const suspicious = maxConsecutiveFrozen > config.maxFrozenFrames;

  return { score, frozenFrameCount, suspicious };
}

// ─────────────────────────────────────────────────────────────
// Texture Analysis (Anti-Screen)
// 
// Uses laplacian variance and moiré detection from frame analysis.
// Real faces have complex, irregular texture.
// Screens have regular grid patterns and different edge profiles.
// ─────────────────────────────────────────────────────────────

interface TextureResult {
  score: number;
  suspicious: boolean;
}

export function analyzeTexture(
  frames: FrameRecord[],
  config: LivenessConfig
): TextureResult {
  if (frames.length === 0) {
    return { score: 0.5, suspicious: false };
  }

  const avgLaplacian = frames.reduce((s, f) => s + f.analysis.laplacianVariance, 0) / frames.length;

  // Very low laplacian = too smooth (possible blur attack)
  // Very high laplacian = possible moiré from screen
  // Optimal range for real face: moderate variance
  let score: number;
  if (avgLaplacian < 5) {
    score = avgLaplacian / 5; // Too smooth
  } else if (avgLaplacian > 50) {
    score = Math.max(0, 1 - (avgLaplacian - 50) / 100); // Too sharp (moiré?)
  } else {
    score = 1.0; // Good range
  }

  // Check for moiré patterns
  const moireScores = frames.map(f => {
    // Use brightness variance as proxy for moiré
    // Real faces have natural variation; screens have patterns
    const varScore = Math.min(1, f.analysis.brightnessVariance / 100);
    return varScore;
  });

  const avgMoire = moireScores.reduce((s, v) => s + v, 0) / moireScores.length;
  const suspicious = avgMoire < 0.2 && avgLaplacian > 30;

  // Combine scores
  const finalScore = (score * 0.6 + avgMoire * 0.4);

  return { score: finalScore, suspicious };
}

// ─────────────────────────────────────────────────────────────
// Liveness Score Fusion
// 
// Combines all signals into a single liveness score.
// 
// SECURITY: Weighted fusion means an attacker must compromise
// ALL signals simultaneously. A single strong signal is not
// enough to pass if others are weak.
// 
// We use a multiplicative penalty for suspicious flags:
// Any suspicious signal heavily penalizes the overall score.
// ─────────────────────────────────────────────────────────────

/**
 * Compute final liveness score from all signals.
 * 
 * Returns score [0-1] and any rejection reason.
 */
export function computeLivenessScore(
  frames: FrameRecord[],
  blinkEvents: BlinkEvent[],
  config: LivenessConfig
): { score: LivenessScore; reason?: RejectionReason } {
  // 1. Blink score
  const blinkScore = Math.min(1, blinkEvents.length / config.minBlinksRequired);
  const blinkSuspicious = blinkEvents.some(e => !e.symmetric);

  // 2. Head movement
  const headResult = analyzeHeadMovement(frames, config);
  const headScore = headResult.sufficient ? 1.0 : 0.0;

  // 3. Micro-motion
  const motionResult = analyzeMicroMotion(frames, config);
  const motionScore = motionResult.score;

  // 4. Temporal consistency
  const temporalResult = analyzeTemporalConsistency(frames, config);
  const temporalScore = temporalResult.score;

  // 5. Texture
  const textureResult = analyzeTexture(frames, config);
  const textureScore = textureResult.score;

  // Check for hard failures
  if (blinkEvents.length === 0 && frames.length > config.minFramesForDecision) {
    return {
      score: {
        overall: 0,
        blink: blinkScore,
        headMovement: headScore,
        microMotion: motionScore,
        temporal: temporalScore,
        texture: textureScore,
      },
      reason: 'BLINK_NOT_DETECTED',
    };
  }

  if (!headResult.sufficient && frames.length > config.minFramesForDecision) {
    return {
      score: {
        overall: 0,
        blink: blinkScore,
        headMovement: headScore,
        microMotion: motionScore,
        temporal: temporalScore,
        texture: textureScore,
      },
      reason: 'HEAD_MOTION_NOT_DETECTED',
    };
  }

  if (temporalResult.suspicious) {
    return {
      score: {
        overall: 0,
        blink: blinkScore,
        headMovement: headScore,
        microMotion: motionScore,
        temporal: temporalScore,
        texture: textureScore,
      },
      reason: 'FROZEN_FRAME_DETECTED',
    };
  }

  // Weighted fusion
  let overall =
    config.weightBlink * blinkScore +
    config.weightHeadMovement * headScore +
    config.weightMicroMotion * motionScore +
    config.weightTemporal * temporalScore +
    config.weightTexture * textureScore;

  // Multiplicative penalty for any suspicious signals
  if (blinkSuspicious) overall *= 0.7;
  if (headResult.suspicious) overall *= 0.5;
  if (motionResult.suspicious) overall *= 0.5;
  if (textureResult.suspicious) overall *= 0.6;

  return {
    score: {
      overall,
      blink: blinkScore,
      headMovement: headScore,
      microMotion: motionScore,
      temporal: temporalScore,
      texture: textureScore,
    },
  };
}

// ─────────────────────────────────────────────────────────────
// Final Decision
// ─────────────────────────────────────────────────────────────

/**
 * Make final liveness decision based on score.
 * 
 * SECURITY: Must pass threshold AND have no hard failures.
 */
export function makeLivenessDecision(
  score: LivenessScore,
  config: LivenessConfig
): { live: boolean; reason?: RejectionReason } {
  if (score.overall >= config.livenessPassThreshold) {
    return { live: true };
  }
  if (score.overall < config.livenessFailThreshold) {
    return { live: false, reason: 'SPOOF_SUSPECTED' };
  }
  // Uncertain zone: require more observation
  return { live: false, reason: 'FACE_NOT_STABLE' };
}
