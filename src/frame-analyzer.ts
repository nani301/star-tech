/**
 * Frame-Level Analysis Module
 * 
 * Analyzes raw video frames for environmental and anti-spoof signals.
 * All operations use Canvas 2D API (no WebGL/ML required).
 * 
 * SECURITY: This is the first line of defense against:
 * - Frozen frames (paused video)
 * - Screen replay (phone/tablet showing photo/video)
 * - Low-light attacks (hiding details)
 * - Uniform images (printed photos on flat surface)
 */

import { FrameAnalysis, LivenessConfig } from './types';

// ─────────────────────────────────────────────────────────────
// Brightness Analysis
// 
// Detects:
// - Too dark (hiding spoof artifacts)
// - Too bright (overexposed, washing out details)
// - Uniform brightness (flat photo, screen)
// ─────────────────────────────────────────────────────────────

/**
 * Analyze brightness of a video frame.
 * 
 * Uses a downsampled version for performance (every 4th pixel).
 * Returns average brightness and variance.
 * 
 * SECURITY: Low variance indicates uniform lighting which is
 * characteristic of digital screens and printed photos.
 * Real faces have natural lighting variation.
 */
export function analyzeBrightness(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): { brightness: number; variance: number } {
  const width = canvas.width;
  const height = canvas.height;

  // Get image data
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  let sum = 0;
  let sumSq = 0;
  let count = 0;

  // Sample every 4th pixel for performance (still ~50k samples for 640x480)
  const step = 4;
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      // Convert RGB to perceived luminance
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      sum += lum;
      sumSq += lum * lum;
      count++;
    }
  }

  const mean = sum / count;
  const variance = (sumSq / count) - (mean * mean);

  return { brightness: mean, variance };
}

// ─────────────────────────────────────────────────────────────
// Frozen Frame Detection
// 
// Compares current frame to previous frame pixel-by-pixel.
// Real video has continuous change; frozen frames are identical.
// 
// SECURITY: This catches:
// - Paused video replay
// - Still photos
// - Frame injection attacks (repeated frames)
// ─────────────────────────────────────────────────────────────

let previousFrameData: ImageData | null = null;

/**
 * Reset frozen frame detector.
 * Call when starting new verification session.
 */
export function resetFrozenFrameDetector(): void {
  previousFrameData = null;
}

/**
 * Compute mean absolute pixel difference between current and previous frame.
 * 
 * Returns 0 for identical frames, higher values for motion.
 * Uses grayscale comparison for efficiency.
 */
export function computeFrameDifference(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): number {
  const width = canvas.width;
  const height = canvas.height;
  const currentData = ctx.getImageData(0, 0, width, height);

  if (!previousFrameData) {
    previousFrameData = currentData;
    return 255; // First frame: assume high difference
  }

  const prev = previousFrameData.data;
  const curr = currentData.data;

  let diffSum = 0;
  let count = 0;
  const step = 4; // Sample every 4th pixel

  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      // Compare luminance
      const prevLum = 0.299 * prev[idx] + 0.587 * prev[idx + 1] + 0.114 * prev[idx + 2];
      const currLum = 0.299 * curr[idx] + 0.587 * curr[idx + 1] + 0.114 * curr[idx + 2];
      diffSum += Math.abs(currLum - prevLum);
      count++;
    }
  }

  // Store current for next comparison
  previousFrameData = currentData;

  return diffSum / count;
}

// ─────────────────────────────────────────────────────────────
// Laplacian Variance (Sharpness / Moiré Detection)
// 
// Laplacian variance measures edge sharpness.
// 
// SECURITY APPLICATIONS:
// - Very low variance: blurry/out-of-focus (possible distant photo)
// - Very high variance with grid pattern: moiré from screen
// - Moderate variance: natural face texture
// 
// Moiré patterns from screens create regular high-frequency grids
// that increase laplacian variance in an unnatural way.
// ─────────────────────────────────────────────────────────────

/**
 * Compute Laplacian variance of the frame.
 * 
 * Uses a simple 3x3 Laplacian kernel approximation:
 *   [ 0 -1  0]
 *   [-1  4 -1]
 *   [ 0 -1  0]
 * 
 * High variance = sharp edges (potential moiré)
 * Low variance = blurry/smooth
 */
export function computeLaplacianVariance(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): number {
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  let sum = 0;
  let sumSq = 0;
  let count = 0;

  // Skip border pixels
  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const idx = (y * width + x) * 4;
      const top = (y - 1) * width + x;
      const bottom = (y + 1) * width + x;
      const left = y * width + (x - 1);
      const right = y * width + (x + 1);

      // Grayscale Laplacian
      const center = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      const t = 0.299 * data[top * 4] + 0.587 * data[top * 4 + 1] + 0.114 * data[top * 4 + 2];
      const b = 0.299 * data[bottom * 4] + 0.587 * data[bottom * 4 + 1] + 0.114 * data[bottom * 4 + 2];
      const l = 0.299 * data[left * 4] + 0.587 * data[left * 4 + 1] + 0.114 * data[left * 4 + 2];
      const r = 0.299 * data[right * 4] + 0.587 * data[right * 4 + 1] + 0.114 * data[right * 4 + 2];

      const laplacian = 4 * center - t - b - l - r;

      sum += laplacian;
      sumSq += laplacian * laplacian;
      count++;
    }
  }

  const mean = sum / count;
  const variance = (sumSq / count) - (mean * mean);

  // Normalize to [0, ~1000] range
  return Math.abs(variance) / 100;
}

// ─────────────────────────────────────────────────────────────
// Moiré Pattern Detection
// 
// Screens create regular grid patterns when captured by cameras.
// We detect this by analyzing the frequency domain.
// 
// Simplified approach (no FFT):
// - Check for periodic intensity variations in rows/columns
// - Combine with laplacian variance spike
// - Check for unnatural edge regularity
// 
// SECURITY: This catches phone/tablet screen replay attacks.
// ─────────────────────────────────────────────────────────────

/**
 * Detect moiré patterns indicative of screen capture.
 * 
 * Returns score [0-1] where higher = more likely screen.
 */
export function detectMoirePattern(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): number {
  const width = canvas.width;
  const height = canvas.height;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  // Analyze horizontal scanlines for periodic patterns
  const rowVariances: number[] = [];
  const stepY = Math.max(1, Math.floor(height / 50)); // Sample ~50 rows

  for (let y = 0; y < height; y += stepY) {
    let rowSum = 0;
    let rowSumSq = 0;
    const stepX = Math.max(1, Math.floor(width / 100));

    for (let x = 0; x < width; x += stepX) {
      const idx = (y * width + x) * 4;
      const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
      rowSum += lum;
      rowSumSq += lum * lum;
    }

    const count = Math.ceil(width / stepX);
    const mean = rowSum / count;
    const variance = (rowSumSq / count) - (mean * mean);
    rowVariances.push(variance);
  }

  // Compute variance OF variances (periodic rows have similar variance)
  const meanVar = rowVariances.reduce((a, b) => a + b, 0) / rowVariances.length;
  const varOfVars = rowVariances.reduce((sum, v) => sum + (v - meanVar) ** 2, 0) / rowVariances.length;

  // Also check for periodic autocorrelation
  let periodicScore = 0;
  if (rowVariances.length > 10) {
    for (let lag = 2; lag < 8; lag++) {
      let corr = 0;
      for (let i = 0; i < rowVariances.length - lag; i++) {
        corr += Math.abs(rowVariances[i] - rowVariances[i + lag]);
      }
      const avgCorr = corr / (rowVariances.length - lag);
      // Low difference at some lag = periodic
      if (avgCorr < meanVar * 0.3) {
        periodicScore += 1;
      }
    }
  }

  // Combine signals
  const regularityScore = Math.max(0, 1 - (varOfVars / 500));
  const periodicityScore = Math.min(1, periodicScore / 3);

  return (regularityScore * 0.5 + periodicityScore * 0.5);
}

// ─────────────────────────────────────────────────────────────
// Comprehensive Frame Analysis
// ─────────────────────────────────────────────────────────────

/**
 * Run all frame-level analyses.
 * 
 * SECURITY: This is called on EVERY frame before any ML inference.
 * If frame analysis fails, we skip expensive model inference.
 */
export function analyzeFrame(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
): FrameAnalysis {
  const { brightness, variance: brightnessVariance } = analyzeBrightness(canvas, ctx);
  const frameDiff = computeFrameDifference(canvas, ctx);
  const laplacianVariance = computeLaplacianVariance(canvas, ctx);

  return {
    brightness,
    brightnessVariance,
    frameDiff,
    laplacianVariance,
    timestamp: performance.now(),
  };
}

// ─────────────────────────────────────────────────────────────
// Environment Validation
// 
// Check if frame passes basic environmental requirements.
// Fail fast to avoid wasting ML compute on bad frames.
// ─────────────────────────────────────────────────────────────

/**
 * Validate frame environment.
 * 
 * Returns { valid: true } or { valid: false, reason }.
 * 
 * SECURITY: Early rejection prevents:
 * - Attacks in very dark rooms (hiding spoof details)
 * - Attacks with overexposed images
 * - Flat uniform images (printed photos)
 */
export function validateEnvironment(
  analysis: FrameAnalysis,
  config: LivenessConfig
): { valid: boolean; reason?: string } {
  if (analysis.brightness < config.minBrightness) {
    return { valid: false, reason: 'LOW_LIGHT' };
  }
  if (analysis.brightness > config.maxBrightness) {
    return { valid: false, reason: 'BRIGHTNESS_TOO_HIGH' };
  }
  if (analysis.brightnessVariance < config.minBrightnessVariance) {
    return { valid: false, reason: 'SPOOF_SUSPECTED' };
  }
  return { valid: true };
}
