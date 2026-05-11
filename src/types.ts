/**
 * Face Liveness Detection - Type Definitions
 * Government-grade browser-side liveness verification
 */

// ─────────────────────────────────────────────────────────────
// Verification Status Enum
// ─────────────────────────────────────────────────────────────
export type VerificationStatus =
  | 'IDLE'
  | 'INITIALIZING'
  | 'ALIGNING'
  | 'OBSERVING'
  | 'BLINK_CHALLENGE'
  | 'HEAD_MOVE_CHALLENGE'
  | 'VERIFYING'
  | 'LIVE_FACE_CONFIRMED'
  | 'FACE_MISMATCH'
  | 'SPOOF_SUSPECTED'
  | 'TIMEOUT'
  | 'ERROR';

// ─────────────────────────────────────────────────────────────
// Rejection Reason Codes
// Each failure gets a specific reason for audit logging
// ─────────────────────────────────────────────────────────────
export type RejectionReason =
  | 'NO_FACE'
  | 'FACE_TOO_SMALL'
  | 'FACE_NOT_STABLE'
  | 'BLINK_NOT_DETECTED'
  | 'HEAD_MOTION_NOT_DETECTED'
  | 'LOW_LIGHT'
  | 'SPOOF_SUSPECTED'
  | 'FACE_MISMATCH'
  | 'FROZEN_FRAME_DETECTED'
  | 'TIMEOUT_EXCEEDED'
  | 'MULTIPLE_FACES_DETECTED'
  | 'BRIGHTNESS_TOO_HIGH'
  | 'UNNATURAL_BLINK_PATTERN'
  | 'REPLAY_ATTACK_DETECTED';

// ─────────────────────────────────────────────────────────────
// User-Facing Prompt Messages
// ─────────────────────────────────────────────────────────────
export type UserPrompt =
  | 'Align your face in the frame'
  | 'Look at the camera'
  | 'Blink once naturally'
  | 'Move your head slightly'
  | 'Stay still, verifying...'
  | 'Live face confirmed'
  | 'Verification failed'
  | 'Face too far'
  | 'Too dark'
  | 'Too bright'
  | 'Multiple faces detected'
  | 'Keep looking at camera';

// ─────────────────────────────────────────────────────────────
// Configuration Interface
// All security-sensitive thresholds are externalized
// ─────────────────────────────────────────────────────────────
export interface LivenessConfig {
  // ── Face Detection ──
  /** Minimum face detection confidence [0-1] */
  faceDetectionThreshold: number;
  /** Minimum face width as ratio of frame width */
  minFaceSizeRatio: number;
  /** Maximum face width as ratio of frame width */
  maxFaceSizeRatio: number;

  // ── Temporal Observation ──
  /** Minimum observation window in milliseconds */
  observationWindowMs: number;
  /** Minimum frames required before any decision */
  minFramesForDecision: number;
  /** Size of circular frame buffer */
  frameBufferSize: number;
  /** Target FPS for processing */
  targetFps: number;

  // ── Blink Detection ──
  /** EAR threshold below which eye is considered closed */
  earThreshold: number;
  /** Minimum consecutive frames for closed state (anti-instant blink) */
  blinkMinClosedFrames: number;
  /** Maximum consecutive frames for closed state (anti-slow fake) */
  blinkMaxClosedFrames: number;
  /** Minimum frames eye must stay open after blink */
  blinkMinOpenFrames: number;
  /** Number of blinks required */
  minBlinksRequired: number;
  /** Maximum blink duration in ms */
  maxBlinkDurationMs: number;
  /** Minimum blink duration in ms */
  minBlinkDurationMs: number;

  // ── Head Movement ──
  /** Minimum head displacement in pixels */
  minHeadMovementPixels: number;
  /** Maximum allowed frame-to-frame head jump (anti-injection) */
  maxHeadJumpPixels: number;
  /** Minimum variance in head position */
  minHeadVariance: number;
  /** Number of frames to track head position */
  headTrackingWindow: number;

  // ── Frame Stability / Anti-Spoof ──
  /** Pixel difference threshold for frozen frame detection */
  frozenFramePixelThreshold: number;
  /** Maximum consecutive frozen frames allowed */
  maxFrozenFrames: number;
  /** Landmark difference threshold for frozen detection */
  frozenFrameLandmarkThreshold: number;
  /** Moiré pattern detection sensitivity */
  moireDetectionSensitivity: number;

  // ── Brightness / Environment ──
  /** Minimum average brightness [0-255] */
  minBrightness: number;
  /** Maximum average brightness [0-255] */
  maxBrightness: number;
  /** Minimum brightness variance (reject uniform images) */
  minBrightnessVariance: number;

  // ── Identity Matching ──
  /** Face descriptor match threshold (cosine similarity) */
  faceMatchThreshold: number;
  /** Number of descriptors to average for enrollment */
  enrollmentSamples: number;

  // ── Timing & Retries ──
  /** Total verification timeout in ms */
  verificationTimeoutMs: number;
  /** Maximum retry attempts */
  maxRetries: number;
  /** Delay between retries in ms */
  retryDelayMs: number;

  // ── Liveness Fusion ──
  /** Minimum liveness score to pass [0-1] */
  livenessPassThreshold: number;
  /** Score below which is immediate fail */
  livenessFailThreshold: number;
  /** Weight for blink signal in fusion */
  weightBlink: number;
  /** Weight for head movement in fusion */
  weightHeadMovement: number;
  /** Weight for micro-motion in fusion */
  weightMicroMotion: number;
  /** Weight for temporal consistency in fusion */
  weightTemporal: number;
  /** Weight for texture analysis in fusion */
  weightTexture: number;
}

// ─────────────────────────────────────────────────────────────
// Face Detection Result
// ─────────────────────────────────────────────────────────────
export interface FaceDetection {
  /** Bounding box [x, y, width, height] in pixels */
  box: [number, number, number, number];
  /** Detection confidence [0-1] */
  score: number;
}

// ─────────────────────────────────────────────────────────────
// 68-Point Landmark Structure
// Indices follow dlib/face-api.js convention
// ─────────────────────────────────────────────────────────────
export interface FaceLandmarks68 {
  /** Raw 68 (x,y) pairs, flattened: [x0,y0, x1,y1, ...] */
  positions: Float32Array;
  /** Image width landmarks were detected on */
  imageWidth: number;
  /** Image height landmarks were detected on */
  imageHeight: number;
}

/** Landmark indices for facial regions */
export const LANDMARK_INDICES = {
  jaw: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
  rightEyebrow: [17, 18, 19, 20, 21],
  leftEyebrow: [22, 23, 24, 25, 26],
  nose: [27, 28, 29, 30, 31, 32, 33, 34, 35],
  rightEye: [36, 37, 38, 39, 40, 41],
  leftEye: [42, 43, 44, 45, 46, 47],
  outerMouth: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
  innerMouth: [60, 61, 62, 63, 64, 65, 66, 67],
} as const;

// ─────────────────────────────────────────────────────────────
// Frame Analysis Result
// ─────────────────────────────────────────────────────────────
export interface FrameAnalysis {
  /** Average brightness [0-255] */
  brightness: number;
  /** Brightness variance */
  brightnessVariance: number;
  /** Mean absolute frame difference from previous */
  frameDiff: number;
  /** Laplacian variance (sharpness/moiré indicator) */
  laplacianVariance: number;
  /** Timestamp of frame capture */
  timestamp: number;
}

// ─────────────────────────────────────────────────────────────
// Temporal Frame Record (stored in circular buffer)
// ─────────────────────────────────────────────────────────────
export interface FrameRecord {
  /** Frame sequence number */
  frameNum: number;
  /** Detection result */
  detection: FaceDetection | null;
  /** Landmarks if face detected */
  landmarks: FaceLandmarks68 | null;
  /** Frame analysis */
  analysis: FrameAnalysis;
  /** Computed EAR (average of both eyes) */
  ear: number;
  /** Head pose estimate */
  headPose: HeadPose;
  /** Nose tip position for tracking */
  nosePosition: { x: number; y: number };
  /** Mouth aspect ratio */
  mar: number;
}

// ─────────────────────────────────────────────────────────────
// Head Pose Estimate (approximate from landmarks)
// ─────────────────────────────────────────────────────────────
export interface HeadPose {
  /** Approximate pitch in degrees */
  pitch: number;
  /** Approximate yaw in degrees */
  yaw: number;
  /** Approximate roll in degrees */
  roll: number;
}

// ─────────────────────────────────────────────────────────────
// Blink Event
// ─────────────────────────────────────────────────────────────
export interface BlinkEvent {
  /** Frame when blink started */
  startFrame: number;
  /** Frame when blink ended */
  endFrame: number;
  /** Duration in milliseconds */
  durationMs: number;
  /** Minimum EAR during blink */
  minEar: number;
  /** Whether both eyes blinked together */
  symmetric: boolean;
}

// ─────────────────────────────────────────────────────────────
// Liveness Score Breakdown
// ─────────────────────────────────────────────────────────────
export interface LivenessScore {
  /** Overall score [0-1] */
  overall: number;
  /** Blink signal score */
  blink: number;
  /** Head movement score */
  headMovement: number;
  /** Micro-motion score */
  microMotion: number;
  /** Temporal consistency score */
  temporal: number;
  /** Texture analysis score */
  texture: number;
}

// ─────────────────────────────────────────────────────────────
// Verification Result
// ─────────────────────────────────────────────────────────────
export interface VerificationResult {
  /** Whether verification passed */
  success: boolean;
  /** Final status */
  status: VerificationStatus;
  /** Rejection reason if failed */
  reason?: RejectionReason;
  /** Liveness score breakdown */
  livenessScore?: LivenessScore;
  /** Face match score if identity checked */
  faceMatchScore?: number;
  /** Number of frames processed */
  framesProcessed: number;
  /** Duration in milliseconds */
  durationMs: number;
  /** Blink events detected */
  blinksDetected: number;
  /** Audit log of state transitions */
  auditLog: AuditEntry[];
}

// ─────────────────────────────────────────────────────────────
// Audit Entry (for security logging)
// ─────────────────────────────────────────────────────────────
export interface AuditEntry {
  timestamp: number;
  status: VerificationStatus;
  reason?: RejectionReason;
  score?: number;
  message: string;
}

// ─────────────────────────────────────────────────────────────
// Callback Types
// ─────────────────────────────────────────────────────────────
export type StatusCallback = (status: VerificationStatus, prompt: UserPrompt) => void;
export type ProgressCallback = (progress: number, message: string) => void;
export type ResultCallback = (result: VerificationResult) => void;

// ─────────────────────────────────────────────────────────────
// Face Descriptor (128-dim from recognition model)
// ─────────────────────────────────────────────────────────────
export type FaceDescriptor = Float32Array; // length 128
