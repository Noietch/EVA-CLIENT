# Changelog

## 0.2.0

### Added

- Added PICO-first VR/WebXR teleoperation, including the PICO page launcher,
  ADB reverse forwarding, controller retargeting, and collection/RL integration
  for AgiBot G2, ARX X5, and Dual Franka.
- Added dual-arm YAM hardware support with paired follower arms and leader
  teaching handles, direct leader-follower teleoperation, relative HIL, stable
  camera serial binding, and deployment/collection presets.
- Added unified dataset readers and export-time writers for LeRobot v3.0,
  MCAP, and HDF5 formats.
- Added ARX X5 hardware support, including its SDK/environment setup, camera
  and URDF assets, deployment presets, and a public evaluation preset.
- Added the browser dataset dashboard with quality review and SFTP export
  workflows.
- Added safe robot bring-up workflows for robot setup, camera calibration, and
  SDK debugging.

### Fixed

- Hardened collection streams and episode history, including frame alignment,
  state-only episodes, history versioning, and console history interactions.
- Enforced neutral VR controllers before collection or RL activation and after
  reconnects, and aligned the supported VR collection and RL control contracts.
- Finalized unified workflow validation while keeping dataset conversion in the
  export path and preserving configured asynchronous saving behavior.
- Stabilized YAM and Orbbec bring-up with sequential multi-camera startup,
  isolated camera capture, USB reset and warmup handling, calibrated D405
  exposure profiles, and single-instance launch protection.
- Fixed interleaved CAN response handling, dual-leader control mapping, and
  collection scheduling for I2RT hardware.

## 0.1.0

### Added

- Initial EVA-Client release with browser-based deployment, evaluation,
  synchronized replay, and LeRobot data collection across the supported robot
  and transport backends.
- Added camera calibration with ChArUco detection, intrinsic/hand-eye/scene
  extrinsic solving, and per-camera end-effector pixel projections in recorded
  episodes.
- Added synchronous and asynchronous episode save pipelines with camera/state
  alignment and bounded background persistence.
- Added human-in-the-loop rollout intervention, concurrent collection review,
  the RL workspace with value-curve visualization, and synchronized replay.
- Added a headless low-power CLI and a ZMQ control channel for driving console
  workflows from simulators and scripts without the web console.

### Fixed

- Fixed Python 3.10 dependency compatibility and made the open-loop install
  verification command run with the bundled sample dataset.
- Fixed rollout HIL recording and control timing, raw stream alignment, and
  replay URDF/video synchronization.

