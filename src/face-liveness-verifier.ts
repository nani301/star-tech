/**
 * Face Liveness Verifier
 * 
 * Main public API for government-grade face liveness detection.
 * 
 * Usage:
 *   const verifier = new FaceLivenessVerifier();
 *   await verifier.initFaceVerifier(videoElement, canvasElement);
 *   const result = await verifier.startVerification(enrolledDescriptor);
 * 
 * SECURITY ARCHITECTURE:
 * 1. Frame-level environmental checks (brightness, frozen frames)
 * 2. Face detection with size validation
 * 3. 68-point landmark extraction
 * 4. Temporal observation with state machine
 * 5. Active challenge-response (blink + head movement)
 * 6. Multi-signal fusion scoring
 * 7. Identity verification (ONLY after liveness confirmed)
 * 8. Final decision with audit trail
 * 
 * This class is designed to be dropped into any page without
 * external dependencies beyond the face model stack.
 */

import {
  LivenessConfig,
  VerificationStatus,
  RejectionReason,
  UserPrompt,
  FaceDescriptor,
  FrameRecord,
  VerificationResult,
  AuditEntry,
  StatusCallback,
  ProgressCallback,
  ResultCallback,
} from './types';

import { DEFAULT_CONFIG } from './config';

import {
  calculateEAR,
  estimateHeadPose,
  getNosePosition,
  calculateMAR,
  computeFaceSimilarity,
} from './landmark-utils';

import {
  analyzeFrame,
  validateEnvironment,
  resetFrozenFrameDetector,
} from './frame-analyzer';

import {
  processBlinkFrame,
  computeLivenessScore,
  makeLivenessDecision,
} from './liveness-engine';

// ─────────────────────────────────────────────────────────────
// Model Inference Interface
// 
// The verifier accepts model inference functions rather than
// importing face-api.js directly. This makes it:
// 1. Framework-agnostic
// 2. Testable with mocks
// 3. Usable with different model loading strategies
// ─────────────────────────────────────────────────────────────

export interface ModelInference {
  /** Detect faces in image element. Returns array of detections with scores. */
  detectFaces(input: HTMLVideoElement | HTMLCanvasElement): Promise<Array<{
    box: { x: number; y: number; width: number; height: number };
    score: number;
  }>>;

  /** Detect 68-point landmarks for a given face box. */
  detectLandmarks(
    input: HTMLVideoElement | HTMLCanvasElement,
    box: { x: number; y: number; width: number; height: number }
  ): Promise<Float32Array | null>; // 136 values = 68 (x,y) pairs

  /** Compute face descriptor (128-dim) for a given face box. */
  computeDescriptor(
    input: HTMLVideoElement | HTMLCanvasElement,
    box: { x: number; y: number; width: number; height: number }
  ): Promise<Float32Array | null>; // 128 values
}

// ─────────────────────────────────────────────────────────────
// Face Liveness Verifier Class
// ─────────────────────────────────────────────────────────────

export class FaceLivenessVerifier {
  // Configuration
  private config: LivenessConfig;

  // DOM elements
  private videoEl: HTMLVideoElement | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  // Model inference functions
  private models: ModelInference | null = null;

  // State
  private status: VerificationStatus = 'IDLE';
  private frameBuffer: FrameRecord[] = [];
  private frameCount = 0;
  private startTime = 0;
  private auditLog: AuditEntry[] = [];

  // Blink tracking
  private blinkMachine = {
    state: 'OPEN' as const,
    framesInState: 0,
    blinkStartFrame: 0,
    minEarInBlink: 1.0,
    blinkEvents: [] as Array<{
      startFrame: number;
      endFrame: number;
      durationMs: number;
      minEar: number;
      symmetric: boolean;
    }>,
  };

  // Timing
  private animationFrameId: number | null = null;
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private retryCount = 0;

  // Callbacks
  private onStatus: StatusCallback | null = null;
  private onProgress: ProgressCallback | null = null;
  private onResult: ResultCallback | null = null;

  // Resolution for processing (downsample for performance)
  private processWidth = 320;
  private processHeight = 240;

  constructor(config?: Partial<LivenessConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ─────────────────────────────────────────────────────────────
  // PUBLIC API
  // ─────────────────────────────────────────────────────────────

  /**
   * Initialize the verifier with video, canvas, and model inference.
   * 
   * @param videoEl - HTMLVideoElement with active webcam stream
   * @param canvasEl - HTMLCanvasElement for frame processing
   * @param models - Model inference functions
   * @param callbacks - Optional status/progress/result callbacks
   */
  public async initFaceVerifier(
    videoEl: HTMLVideoElement,
    canvasEl: HTMLCanvasElement,
    models: ModelInference,
    callbacks?: {
      onStatus?: StatusCallback;
      onProgress?: ProgressCallback;
      onResult?: ResultCallback;
    }
  ): Promise<void> {
    this.videoEl = videoEl;
    this.canvasEl = canvasEl;
    this.models = models;

    if (callbacks) {
      this.onStatus = callbacks.onStatus || null;
      this.onProgress = callbacks.onProgress || null;
      this.onResult = callbacks.onResult || null;
    }

    // Setup canvas context
    const ctx = canvasEl.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }
    this.ctx = ctx;

    // Set processing resolution
    this.processWidth = Math.min(320, videoEl.videoWidth || 320);
    this.processHeight = Math.min(240, videoEl.videoHeight || 240);
    canvasEl.width = this.processWidth;
    canvasEl.height = this.processHeight;

    this.setStatus('INITIALIZING');
    this.logAudit('INITIALIZING', 'Verifier initialized');
  }

  /**
   * Start a verification session.
   * 
   * SECURITY: This is the main entry point. It runs the full
   * verification pipeline and returns only when:
   * - Liveness is confirmed AND identity matches, OR
   * - Timeout/retry limit reached with rejection
   * 
   * @param enrolledDescriptor - The enrolled face descriptor to match against
   * @returns VerificationResult with full audit trail
   */
  public async startVerification(
    enrolledDescriptor: FaceDescriptor
  ): Promise<VerificationResult> {
    return new Promise((resolve) => {
      if (!this.videoEl || !this.ctx || !this.models) {
        resolve(this.buildResult(false, 'ERROR', 'NOT_INITIALIZED'));
        return;
      }

      this.resetVerification();
      this.startTime = performance.now();
      this.retryCount = 0;

      // Set overall timeout
      this.timeoutId = setTimeout(() => {
        this.stopProcessing();
        resolve(this.buildResult(false, 'TIMEOUT', 'TIMEOUT_EXCEEDED'));
      }, this.config.verificationTimeoutMs);

      // Start processing loop
      this.setStatus('ALIGNING');
      this.logAudit('ALIGNING', 'Starting verification session');

      const processFrame = async () => {
        const result = await this.processSingleFrame(enrolledDescriptor);

        if (result) {
          // Verification complete
          this.stopProcessing();
          if (this.timeoutId) clearTimeout(this.timeoutId);
          resolve(result);
          return;
        }

        // Continue processing
        this.animationFrameId = requestAnimationFrame(processFrame);
      };

      this.animationFrameId = requestAnimationFrame(processFrame);
    });
  }

  /**
   * Check if face is currently live (synchronous check).
   * 
   * Use this for continuous monitoring, not for final decision.
   * Returns current liveness score without identity check.
   */
  public async verifyLiveFace(): Promise<{
    live: boolean;
    score: number;
    reason?: RejectionReason;
  }> {
    // Process a few frames and return quick assessment
    // This is a simplified version for continuous monitoring
    const frames = this.frameBuffer;
    if (frames.length < this.config.minFramesForDecision) {
      return { live: false, score: 0, reason: 'FACE_NOT_STABLE' };
    }

    const { score, reason } = computeLivenessScore(
      frames,
      this.blinkMachine.blinkEvents,
      this.config
    );

    const decision = makeLivenessDecision(score, this.config);
    return {
      live: decision.live,
      score: score.overall,
      reason: decision.reason || reason,
    };
  }

  /** Get current verification status */
  public getVerificationStatus(): VerificationStatus {
    return this.status;
  }

  /** Reset verifier state for new session */
  public resetVerification(): void {
    this.stopProcessing();
    this.frameBuffer = [];
    this.frameCount = 0;
    this.blinkMachine = {
      state: 'OPEN',
      framesInState: 0,
      blinkStartFrame: 0,
      minEarInBlink: 1.0,
      blinkEvents: [],
    };
    this.auditLog = [];
    this.retryCount = 0;
    resetFrozenFrameDetector();
    this.setStatus('IDLE');
  }

  /** Clean up resources */
  public destroy(): void {
    this.stopProcessing();
    this.videoEl = null;
    this.canvasEl = null;
    this.ctx = null;
    this.models = null;
  }

  // ─────────────────────────────────────────────────────────────
  // PRIVATE: Frame Processing Pipeline
  // ─────────────────────────────────────────────────────────────

  /**
   * Process a single frame through the full pipeline.
   * 
   * SECURITY: This is the core per-frame logic. Every frame goes through:
   * 1. Environmental validation
   * 2. Face detection
   * 3. Landmark extraction
   * 4. Temporal signal computation
   * 5. State machine updates
   * 6. Decision evaluation
   */
  private async processSingleFrame(
    enrolledDescriptor: FaceDescriptor
  ): Promise<VerificationResult | null> {
    if (!this.videoEl || !this.ctx || !this.models) return null;

    // Skip if video not ready
    if (this.videoEl.readyState < 2) return null;

    this.frameCount++;

    // ── Step 1: Capture frame to canvas ──
    this.ctx.drawImage(this.videoEl, 0, 0, this.processWidth, this.processHeight);

    // ── Step 2: Frame-level analysis (brightness, frozen frame, etc.) ──
    const frameAnalysis = analyzeFrame(this.canvasEl, this.ctx);
    const envCheck = validateEnvironment(frameAnalysis, this.config);

    if (!envCheck.valid) {
      this.handleEnvironmentFailure(envCheck.reason!);
      return null;
    }

    // ── Step 3: Face detection ──
    const detections = await this.models.detectFaces(this.videoEl);

    if (detections.length === 0) {
      this.handleNoFace();
      return null;
    }

    if (detections.length > 1) {
      // Multiple faces detected - security risk
      this.setStatus('SPOOF_SUSPECTED');
      this.logAudit('SPOOF_SUSPECTED', 'MULTIPLE_FACES_DETECTED');
      return this.buildResult(false, 'SPOOF_SUSPECTED', 'MULTIPLE_FACES_DETECTED');
    }

    const detection = detections[0];

    // Validate face size
    const faceSizeRatio = detection.box.width / this.processWidth;
    if (faceSizeRatio < this.config.minFaceSizeRatio) {
      this.setStatus('ALIGNING');
      this.emitPrompt('Face too far');
      return null;
    }
    if (faceSizeRatio > this.config.maxFaceSizeRatio) {
      this.setStatus('ALIGNING');
      this.emitPrompt('Align your face in the frame');
      return null;
    }

    // ── Step 4: Landmark detection ──
    const landmarksRaw = await this.models.detectLandmarks(this.videoEl, detection.box);
    if (!landmarksRaw) {
      this.handleNoFace();
      return null;
    }

    const landmarks = {
      positions: landmarksRaw,
      imageWidth: this.processWidth,
      imageHeight: this.processHeight,
    };

    // ── Step 5: Compute physiological signals ──
    const earResult = calculateEAR(landmarks);
    const headPose = estimateHeadPose(landmarks);
    const nosePos = getNosePosition(landmarks);
    const mar = calculateMAR(landmarks);

    // ── Step 6: Build frame record ──
    const record: FrameRecord = {
      frameNum: this.frameCount,
      detection: {
        box: [detection.box.x, detection.box.y, detection.box.width, detection.box.height],
        score: detection.score,
      },
      landmarks,
      analysis: frameAnalysis,
      ear: earResult.average,
      headPose,
      nosePosition: nosePos,
      mar,
    };

    // Add to circular buffer
    this.frameBuffer.push(record);
    if (this.frameBuffer.length > this.config.frameBufferSize) {
      this.frameBuffer.shift();
    }

    // ── Step 7: Update state machine ──
    this.updateStateMachine(record, earResult.symmetric);

    // ── Step 8: Evaluate decision ──
    return this.evaluateDecision(enrolledDescriptor);
  }

  // ─────────────────────────────────────────────────────────────
  // PRIVATE: State Machine & Decision Logic
  // ─────────────────────────────────────────────────────────────

  /**
   * Update verification state based on current frame.
   * 
   * State progression:
   * ALIGNING → OBSERVING → BLINK_CHALLENGE → HEAD_MOVE_CHALLENGE → VERIFYING
   */
  private updateStateMachine(record: FrameRecord, eyeSymmetric: boolean): void {
    const elapsed = performance.now() - this.startTime;
    const bufferSize = this.frameBuffer.length;

    switch (this.status) {
      case 'ALIGNING':
        if (bufferSize >= 10) {
          this.setStatus('OBSERVING');
          this.emitPrompt('Look at the camera');
        }
        break;

      case 'OBSERVING':
        // Wait for minimum observation window
        if (elapsed >= this.config.observationWindowMs && bufferSize >= this.config.minFramesForDecision) {
          this.setStatus('BLINK_CHALLENGE');
          this.emitPrompt('Blink once naturally');
        }
        break;

      case 'BLINK_CHALLENGE': {
        // Process blink detection
        const blinkResult = processBlinkFrame(this.blinkMachine, record, this.config);

        // Add any detected blink events
        for (const event of blinkResult.events) {
          event.symmetric = eyeSymmetric;
          this.blinkMachine.blinkEvents.push(event);
        }

        if (blinkResult.suspicious) {
          this.logAudit('SPOOF_SUSPECTED', 'UNNATURAL_BLINK_PATTERN');
        }

        // If enough blinks detected, move to head movement
        if (this.blinkMachine.blinkEvents.length >= this.config.minBlinksRequired) {
          this.setStatus('HEAD_MOVE_CHALLENGE');
          this.emitPrompt('Move your head slightly');
        }
        break;
      }

      case 'HEAD_MOVE_CHALLENGE': {
        // Check for head movement
        const headResult = analyzeHeadMovement(this.frameBuffer, this.config);
        if (headResult.sufficient) {
          this.setStatus('VERIFYING');
          this.emitPrompt('Stay still, verifying...');
        }
        break;
      }

      case 'VERIFYING':
        // Continue accumulating frames for robust fusion
        break;
    }
  }

  /**
   * Evaluate whether we have enough evidence for a decision.
   * 
   * SECURITY: This is the critical decision point.
   * We require:
   * 1. Minimum observation window elapsed
   * 2. Minimum frames processed
   * 3. Liveness score above threshold
   * 4. Face identity match
   * 
   * BOTH must pass. Either failing = rejection.
   */
  private async evaluateDecision(
    enrolledDescriptor: FaceDescriptor
  ): Promise<VerificationResult | null> {
    const elapsed = performance.now() - this.startTime;

    // Not enough data yet
    if (this.status !== 'VERIFYING' && this.status !== 'HEAD_MOVE_CHALLENGE') {
      return null;
    }

    // Need minimum frames
    if (this.frameBuffer.length < this.config.minFramesForDecision) {
      return null;
    }

    // Need minimum time
    if (elapsed < this.config.observationWindowMs) {
      return null;
    }

    // ── Compute liveness score ──
    const { score: livenessScore, reason: livenessReason } = computeLivenessScore(
      this.frameBuffer,
      this.blinkMachine.blinkEvents,
      this.config
    );

    const livenessDecision = makeLivenessDecision(livenessScore, this.config);

    if (!livenessDecision.live) {
      // Liveness failed - do NOT check identity
      this.setStatus('SPOOF_SUSPECTED');
      this.logAudit('SPOOF_SUSPECTED', livenessDecision.reason || 'SPOOF_SUSPECTED');
      return this.buildResult(
        false,
        'SPOOF_SUSPECTED',
        livenessDecision.reason || 'SPOOF_SUSPECTED',
        livenessScore
      );
    }

    // ── Liveness passed! Now check identity ──
    this.setStatus('VERIFYING');

    // Extract descriptor from current frame
    const lastRecord = this.frameBuffer[this.frameBuffer.length - 1];
    if (!lastRecord.detection) {
      return this.buildResult(false, 'ERROR', 'NO_FACE');
    }

    const [x, y, w, h] = lastRecord.detection.box;
    const currentDescriptor = await this.models!.computeDescriptor(this.videoEl!, {
      x, y, width: w, height: h,
    });

    if (!currentDescriptor) {
      return this.buildResult(false, 'ERROR', 'NO_FACE');
    }

    const similarity = computeFaceSimilarity(currentDescriptor, enrolledDescriptor);

    if (similarity >= this.config.faceMatchThreshold) {
      // BOTH passed!
      this.setStatus('LIVE_FACE_CONFIRMED');
      this.emitPrompt('Live face confirmed');
      this.logAudit('LIVE_FACE_CONFIRMED', `Match score: ${similarity.toFixed(3)}`);
      return this.buildResult(true, 'LIVE_FACE_CONFIRMED', undefined, livenessScore, similarity);
    } else {
      // Liveness passed but identity mismatch
      this.setStatus('FACE_MISMATCH');
      this.emitPrompt('Verification failed');
      this.logAudit('FACE_MISMATCH', `Match score: ${similarity.toFixed(3)}`);
      return this.buildResult(false, 'FACE_MISMATCH', 'FACE_MISMATCH', livenessScore, similarity);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // PRIVATE: Event Handlers
  // ─────────────────────────────────────────────────────────────

  private handleNoFace(): void {
    // Don't immediately fail - face might just be temporarily lost
    // But if we've been in ALIGNING too long, eventually timeout handles it
    this.emitPrompt('Align your face in the frame');
  }

  private handleEnvironmentFailure(reason: string): void {
    switch (reason) {
      case 'LOW_LIGHT':
        this.emitPrompt('Too dark');
        break;
      case 'BRIGHTNESS_TOO_HIGH':
        this.emitPrompt('Too bright');
        break;
      case 'SPOOF_SUSPECTED':
        this.emitPrompt('Verification failed');
        break;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // PRIVATE: Utilities
  // ─────────────────────────────────────────────────────────────

  private setStatus(status: VerificationStatus): void {
    if (this.status === status) return;
    this.status = status;

    const prompt = this.statusToPrompt(status);
    if (this.onStatus) {
      this.onStatus(status, prompt);
    }
  }

  private emitPrompt(prompt: UserPrompt): void {
    // Progress callback can also receive prompts
    if (this.onProgress) {
      const progress = this.calculateProgress();
      this.onProgress(progress, prompt);
    }
  }

  private statusToPrompt(status: VerificationStatus): UserPrompt {
    switch (status) {
      case 'ALIGNING': return 'Align your face in the frame';
      case 'OBSERVING': return 'Look at the camera';
      case 'BLINK_CHALLENGE': return 'Blink once naturally';
      case 'HEAD_MOVE_CHALLENGE': return 'Move your head slightly';
      case 'VERIFYING': return 'Stay still, verifying...';
      case 'LIVE_FACE_CONFIRMED': return 'Live face confirmed';
      case 'FACE_MISMATCH': return 'Verification failed';
      case 'SPOOF_SUSPECTED': return 'Verification failed';
      case 'TIMEOUT': return 'Verification failed';
      default: return 'Align your face in the frame';
    }
  }

  private calculateProgress(): number {
    const elapsed = performance.now() - this.startTime;
    const progress = Math.min(100, (elapsed / this.config.observationWindowMs) * 100);
    return progress;
  }

  private logAudit(status: VerificationStatus, message: string, reason?: RejectionReason): void {
    this.auditLog.push({
      timestamp: performance.now(),
      status,
      reason,
      message,
    });
  }

  private stopProcessing(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.timeoutId !== null) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }

  private buildResult(
    success: boolean,
    status: VerificationStatus,
    reason?: RejectionReason,
    livenessScore?: { overall: number; blink: number; headMovement: number; microMotion: number; temporal: number; texture: number },
    faceMatchScore?: number
  ): VerificationResult {
    const duration = performance.now() - this.startTime;

    return {
      success,
      status,
      reason,
      livenessScore: livenessScore ? {
        overall: livenessScore.overall,
        blink: livenessScore.blink,
        headMovement: livenessScore.headMovement,
        microMotion: livenessScore.microMotion,
        temporal: livenessScore.temporal,
        texture: livenessScore.texture,
      } : undefined,
      faceMatchScore,
      framesProcessed: this.frameCount,
      durationMs: duration,
      blinksDetected: this.blinkMachine.blinkEvents.length,
      auditLog: [...this.auditLog],
    };
  }
}

// Re-export types for convenience
export * from './types';
export * from './config';
export { DEFAULT_CONFIG, DEV_CONFIG, ULTRA_SECURE_CONFIG } from './config';
