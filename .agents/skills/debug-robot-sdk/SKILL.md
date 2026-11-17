---
name: debug-robot-sdk
description: Diagnose and validate a physical robot SDK from offline import through read-only telemetry, motor enable, gripper probing, and bounded joint motion. Use for new SDK bring-up, connection failures, command-path verification, or live hardware acceptance; every device, motor-enable, gripper, and motion stage requires explicit user confirmation.
---

# Debug Robot SDK

Treat offline import, live connection, motor enable, each gripper, and each arm motion as separate stages. Permission for one stage, side, or device never carries to another.

## Establish The Target

Collect or verify:

- robot identity, controller, SDK source and revision, environment, and executor;
- joint order, SDK index, units, limits, command mode, speed and torque bounds;
- exact device, controller, CAN, serial, ROS topic, or endpoint;
- emergency stop, workspace clearance, brakes, support, and shutdown procedure;
- whether the agent will execute commands or the user will execute them.

For a dual-arm robot, ask the user to confirm physical left/right arm to SDK channel/index, controller or CAN/device, and EVA actuator group. For cameras involved in observation, confirm physical left/right or scene position to serial/index and EVA key. Do not continue with ambiguous mappings.

For each gripper, separately ask whether it is currently `open`, `closed`, or `unknown`. Record the SDK value-to-physical-opening convention; do not infer physical state from a raw number until that convention is confirmed.

Always state angular values in radians and degrees.

## Stage 1: Offline Validation

Import the SDK, inspect capabilities and signatures, validate configuration and joint mappings, and exercise mocks or fake nodes. Do not open a live device. Report offline results before requesting live access.

## Stage 2: Live Connection And Read-Only Telemetry

Before connecting, tell the user the exact robot, side, device or endpoint, connection timeout, operations to be performed, expected no-motion state, and cleanup behavior. Ask for explicit confirmation.

Keep motors disabled. Use a bounded connection attempt, read identity and telemetry, compare reported joints with the confirmed mapping, and disconnect. Do not clear faults, enable torque, or send commands. Any retry requires a new confirmation.

## Stage 3: Motor Enable

Before enabling motors, state the exact robot and physical side, SDK channel, joints affected, expected brake or holding behavior, timeout, executor, abort action, and how motors will be disabled. Ask for explicit confirmation.

Enable only the confirmed side and observe without commanding motion. Disable immediately on unexpected movement, fault, stale telemetry, mapping mismatch, or user stop. Do not silently clear a fault or retry.

## Stage 4: Gripper Probe

Debug each gripper independently. Reconfirm whether that physical gripper is open, closed, or unknown immediately before a command. If unknown, use only a separately confirmed read-only observation or camera check to establish state.

Before a gripper command, state:

- robot, physical left/right side, SDK channel, and EVA group;
- confirmed current physical state and SDK value;
- exact target SDK value and physical opening or closing distance;
- speed, force, torque, or current limit;
- expected contact behavior, return behavior, timeout, executor, and abort action.

Ask for explicit confirmation. Do not move an arm during the gripper probe, do not assume the other gripper has the same calibration, and do not reuse authorization for the other side or return command.

## Stage 5: Bounded Arm Motion

Select the smallest meaningful joint delta from the SDK resolution, robot limits, and telemetry noise. Values such as `0.01 rad` (`0.573 deg`), `0.05 rad` (`2.865 deg`), or `1 deg` (`0.01745 rad`) are examples, not defaults.

Before moving, state:

- robot, physical left/right side, SDK channel, EVA group, joint name, and SDK index;
- current angle, direction, exact delta, and absolute target in both radians and degrees;
- speed, acceleration, torque/current limit, position tolerance, timeout, and workspace clearance;
- whether a return motion is proposed, who executes each command, and the exact stop/disable action.

Ask for confirmation of that exact command. If the agent executes it, repeat the values immediately before execution. If the user executes it, provide the exact command and repeat the same values before asking them to run it. A changed angle, joint, side, executor, retry, or return motion requires a new confirmation.

Stop on unexpected direction, excess error, limit proximity, stale telemetry, fault, obstruction, or user instruction. Disable safely and report; do not improvise recovery motion.

## Report Results

For every stage, report whether it was skipped, confirmed, passed, or stopped. Include arm and camera mappings, motor state, the gripper's starting state and target/result, commanded and measured angles in radians and degrees, observed faults, cleanup state, and any remaining live confirmation.
