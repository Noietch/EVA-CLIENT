# Changelog

## 0.2.0

- Added PICO-first VR/WebXR teleoperation with protocol version 3, absolute hand poses, controller retargeting, grip authorization, haptic feedback, and reconnect-aware pose deltas.
- Added PICO-first VR operation: the PICO launcher opens the WebXR page through ADB reverse port forwarding, supports explicit or auto-detected device serials, and documents the SSH tunnel required by the headset. Quest input remains supported by the same WebXR node.
- Added optional `VR_TOKEN` propagation from the launcher to the WebXR node so PICO sessions can be opened with an explicit authenticated URL.
- Added VR retargeting from controller position/quaternion deltas to EEF targets with configurable coordinate rotation, position scale, workspace bounds, low-pass filtering, relative references, and downstream IK/error safety checks.
- Added WebXR operator gestures for record toggle, long-press record cancel, global ARM toggle, HOME, RL intervention toggle, per-hand grip authorization, and event acknowledgements.
- Added VR frame and acknowledgement channels with session/sequence validation, heartbeat monitoring, bounded retry queues, authentication tokens, stale-frame rejection, and fail-closed handling for protocol errors.
- Added VR collection and RL presets for AgiBot G2, ARX X5, and Dual Franka, with explicit support boundaries so unsupported robots are not routed through the VR control path.
- Added dual-arm YAM hardware support with separate left/right Follower arms and paired Leader teaching handles, a registered YAM robot model and URDF, direct Leader-to-Follower teleoperation, relative HIL takeover, and dedicated deployment/collection configurations.
- Added a YAM hardware launcher with the I2RT SDK fork, persistent CAN adapter mapping, automatic gripper calibration, configurable control/publish rates, startup motion, optional bounded tracking trim, and configurable paired or single-Leader operation.
- Added YAM relative collection/HIL anchoring so leader deltas are applied to the current follower pose without zero-offset jumps, while calibrated leader gripper encoder endpoints map to follower grippers in normalized `[0, 1]` space.
- Added an isolated YAM environment and process boundary to keep the I2RT SDK's NumPy 2.x dependencies separate from EVA Client's NumPy 1.x runtime while exchanging observations and actions over ZeroMQ.
- Added YAM camera support for serial-bound Orbbec overhead and wrist cameras, shared isolated multi-camera capture, optional RealSense D405 overhead input, and default three-camera collection wiring.
- Added unified dataset readers and export-time conversion from the existing LeRobot v2.1 episode layout to LeRobot v3.0, MCAP, and HDF5 outputs.
- Added quality-aware dataset export that classifies raw LeRobot v2.1 episodes into accepted and rejected subsets, preserves source episode indices and metadata/statistics, writes `quality_split.json` markers, and publishes both subsets atomically.
- Added LeRobot v3.0 export with Parquet/video sharding, episode/task metadata, schema-aware shard rollover, and frame-count validation.
- Added MCAP export with message-packed episode metadata, non-image columns, inline images, aligned per-camera image messages, timestamps, and sequence numbers.
- Added HDF5 export with separate columns, images, and metadata groups, preservation of scalar/container values, appendable image datasets, and image-shape/frame-count validation.
- Added the ARX X5 robot adapter, official SDK integration, isolated Python environment setup, camera support, reset/fake-run tooling, robot assets, and deployment presets.
- Added a public ARX X5 evaluation preset and RL configuration.
- Added the browser dataset dashboard with raw collection/evaluation discovery, date filters, duration and valid-rate metrics, active-span efficiency, evaluation success rate, task demand, recent episodes, and calendar heatmaps.
- Added collection quality-transfer controls with `lerobot_v3`, `hdf5`, and `mcap` format selection, asynchronous episode/file/byte progress, accepted/rejected summaries, and upload gating that only permits the current accepted export.
- Added configurable SFTP dataset publishing with host/port/user/identity-file settings, absolute remote directories, progress reporting, staging, atomic remote publication, and concurrent-upload protection.
- Added safe robot bring-up workflows for adding robots, calibrating cameras, and debugging robot SDKs.

- Hardened collection streams and episode history with bounded stream handling, camera/frame alignment, state-only episode support, cached history counts, history version invalidation, and robust console history pagination.
- Hardened VR safety around ARM/teleop activation and reconnects by resetting accumulated pose references and per-arm grip permissions, rejecting stale controller generations, and keeping the global ARM gate independent from per-arm grip authorization.
- Finalized the VR collection/RL contracts so only the supported robot presets expose VR clients, collection and RL share the same control queue, and reconnects cannot re-arm motion from stale controller state.
- Finalized unified workflow validation while keeping dataset-format conversion in the export path, preserving configured asynchronous-save semantics, and avoiding recording-time format branches.
- Stabilized YAM bring-up by starting follower/leader interfaces in a verified order, validating the default paired Leader CAN mapping while preserving explicit single-Leader modes, and waiting for camera readiness before enabling motors.
- Isolated all enabled Orbbec streams in one SDK process/context, added USB reset and frame warmup recovery, loaded calibrated D405 exposure profiles, and prevented a second launcher from resetting a live camera session.
- Filtered interleaved I2RT CAN responses, corrected the dual-leader mapping, and stabilized collection scheduling and startup tracking behavior.
- Aligned ARM gating with grip permissions so a disabled ARM cannot expose or accept stale VR grip authorization after reconnects or state changes.
- Clarified collection control feedback by showing ARM-OFF as disabled, exposing long-press progress, making record/cancel state transitions unambiguous, and preventing controls from appearing ready while the gate is off.
- Fixed the Dashboard calendar heatmap legend so LESS/MORE labels remain visible across scrolling and viewport sizes.
- Fixed ARX X5 hardware startup when `PYTHONPATH` is unset or shell strict mode is enabled by initializing the repository and SDK paths safely.

## 0.1.0

- Initial EVA-Client release with browser-based deployment, evaluation, collection, replay, visualization, and a shared configuration/state-machine layer over ROS 1, ROS 2, ZeroMQ, and offline dataset transports.
- Added policy deployment strategies for synchronous, asynchronous, naive, ACT-ensemble, and real-time-chunking inference, with OpenPI, OpenPI-RTC, StarVLA, GR00T, mock, and replay backends.
- Added camera calibration with a ChArUco board workflow, editable pose lists, intrinsic calibration, hand-eye solving, scene extrinsics, atomic raiden-compatible JSON output, printable board PDF export, camera frusta, and per-camera end-effector pixel projections in recorded episodes.
- Added synchronous episode saving with camera/state alignment, LeRobot v2.1 metadata, encoded camera streams, quality flags, and bounded capture buffers.
- Added asynchronous save support for evaluation and rollout episodes, with configurable queues and save behavior shared across console workflows.
- Added rollout evaluation improvements and R1 Lite HIL presets, including smooth action processing, per-trial records, replay metadata, and policy intervention capture.
- Added confirmed human-in-the-loop takeover for supported robots, including fake-hardware coverage for testing and explicit operator confirmation before motion control changes.
- Added concurrent collection review with in-tab episode replay, PASS/FAIL quality decisions, and shared collection/replay state transitions.
- Added the RL workspace with rollout controls, intervention state, critic client integration, value-curve visualization, replay controls, and RL configuration presets for supported robots.
- Added rollout metadata refresh so recorded action keys and episode metadata remain consistent with the active control schema.
- Added streaming rollout video persistence and bounded ROS capture queues to reduce memory growth during long runs.
- Added a headless low-power mode that runs transport, policy, and state-machine inference without the HTTP console, 3D scene, camera streams, or episode preview, while retaining structured status events and a heartbeat.
- Added a ZMQ REP control channel and `eva_ctl` client for allow-listed console commands, collection task/ARM mutations, status/config/frame queries, and simulator/script control without a browser.
- Added dual Piper deployment transport configuration with ROS settings, per-robot image color conversion, and SSH policy-port forwarding.
- Added CI enforcement for Ruff linting/formatting so pull requests detect formatting regressions automatically.

- Fixed Python 3.10 dependency compatibility by pinning compatible JAX versions, and made the open-loop install verification command run with the bundled sample dataset.
- Fixed rollout HIL recording and raw stream alignment so intervention segments and camera observations remain paired with the correct control step.
- Fixed rollout HIL control timing and save ordering for deterministic episode boundaries and reliable asynchronous persistence.
- Fixed replay URDF transforms and video synchronization so the 3D scene and camera timeline stay aligned during review.
