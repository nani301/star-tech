🔐 Face Liveness Detection
Government-grade, browser-side face liveness verification module for anti-spoof identity verification.

What This Is
A reusable, production-grade TypeScript module that plugs into any face-identification flow to verify that the face in front of the camera is a live human, not a photo, screen replay, video, or deepfake.

Security Model

| Threat              | Defense                                                               |
| ------------------- | --------------------------------------------------------------------- |
| Printed photo       | Blink detection + temporal consistency + micro-motion                 |
| Phone/tablet screen | Moiré detection + frozen frame detection + challenge-response         |
| Replay video        | Frame difference analysis + periodicity detection + timing validation |
| Deepfake/Synthetic  | Micro-movement analysis + blink dynamics + multi-signal fusion        |
| 3D mask             | Multi-angle challenge + texture analysis                              |
| Frozen frame        | Pixel-level frame difference + landmark jitter detection              |
| Frame injection     | Smoothness validation + timestamp integrity                           |

Core principle: Liveness and Identity are independently verified. Both must pass.



Architecture

┌─────────────────────────────────────────────────────────────┐
│                    FaceLivenessVerifier                        │
├─────────────────────────────────────────────────────────────┤
│  initFaceVerifier(video, canvas, models, callbacks)           │
│  startVerification(enrolledDescriptor) → Promise<Result>    │
│  verifyLiveFace() → { live, score, reason }                 │
│  getVerificationStatus() → Status                            │
│  resetVerification()                                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌─────────────┐
   │  FRAME  │         │ LANDMARK│         │  LIVENESS   │
   │ ANALYZER│         │  UTILS  │         │   ENGINE    │
   │         │         │         │         │             │
   │Brightness│        │  EAR    │         │Blink State  │
   │Frozen   │         │  MAR    │         │Machine     │
   │Frame    │         │Head Pose│         │Head Move   │
   │Moire    │         │Nose Pos │         │Micro-motion│
   │         │         │         │         │Temporal    │
   │         │         │         │         │Texture     │
   └─────────┘         └─────────┘         └─────────────┘
   

   Files
   
| File                            | Purpose                                                         |
| ------------------------------- | --------------------------------------------------------------- |
| `src/types.ts`                  | TypeScript interfaces, enums, and type definitions              |
| `src/config.ts`                 | Default, dev, and ultra-secure configuration presets            |
| `src/landmark-utils.ts`         | Pure functions: EAR, MAR, head pose, descriptor comparison      |
| `src/frame-analyzer.ts`         | Frame-level analysis: brightness, frozen frame, moiré           |
| `src/liveness-engine.ts`        | Core engine: blink state machine, motion analysis, score fusion |
| `src/face-liveness-verifier.ts` | Main public API class                                           |
| `src/face-api-adapter.ts`       | Adapter for face-api.js model stack                             |
| `src/index.ts`                  | Barrel export                                                   |
| `example-usage.html`            | Complete working example                                        |
| `LIVENESS_ARCHITECTURE.md`      | Deep architectural design document                              |


