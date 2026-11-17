---
name: add-robots
description: Integrate a new physical robot into EVA-CLIENT from an SDK and URDF, including its isolated hardware environment, runtime adapter, configs, fake node, visualization assets, and validation. Use when adding a robot family or bringing a vendor SDK into the repository; do not use for routine changes to an existing adapter.
---

# Add Robots

Build a complete EVA-CLIENT robot adapter without treating repository access as permission to touch live hardware. Keep discovery, offline integration, live connection, camera calibration, and motion as separate stages.

## Establish The Integration Contract

Read the repository `CODING_STYLE.md` and [the repository map](references/repository-map.md) before editing. Inspect nearby adapters and select the closest transport and SDK pattern.

Ask for any missing information that cannot be discovered safely:

1. Robot display name and stable snake-case slug, single- or dual-arm topology, joint order, units, limits, gripper convention, and action/state dimensions.
2. SDK source as either a Git URL plus exact revision or an exact local path. Confirm whether it will be a submodule, copied dependency, or local dependency. Never move or delete the supplied source.
3. URDF path or URL, mesh roots, package URI rules, collision requirements, destination, and redistribution/license constraints.
4. Runtime transport and exact endpoints, CAN interfaces, serial devices, controller addresses, ROS topics, or ZMQ endpoints.
5. Required Python version, native libraries, ROS distribution, environment variables, udev rules, and system packages.
6. Camera SDK, physical mounting position, device serial or stable identifier, EVA observation key, resolution, frame rate, and calibration data.

For every dual-arm robot, require an explicit table mapping physical left/right arm to SDK channel or index, controller/CAN/device, and EVA actuator group. For every multi-camera robot, require an explicit table mapping physical left/right or scene position to serial/index and EVA key such as `cam_left_wrist` or `cam_right_wrist`. Never infer either mapping from enumeration order.

## Inspect Before Implementing

Inspect the SDK without opening live devices. Record:

- import and construction APIs;
- state, command, enable, disable, fault, stop, and shutdown behavior;
- joint ordering, units, limits, and gripper representation;
- blocking calls, callback threads, and cleanup requirements;
- camera controls and capability-query APIs;
- bundled URDF and mesh resources;
- native-library, ROS, permission, or device requirements.

Resolve ambiguous contracts with the user before writing an adapter. Vendor SDK names may remain unchanged even when the EVA-facing robot slug differs.

## Implement The Adapter

Keep the change coherent and follow the selected repository pattern:

1. Add the robot model, registration, URDF, and meshes under `src/robots/zoo/<slug>/`.
2. Create `examples/hardware/<slug>/pyproject.toml` and a local setup script. All entry points must use that directory's `.venv`; do not add robot-specific dependencies or extras to the root environment.
3. Add the hardware node, SDK wrapper, robot adapter, camera adapter when applicable, deterministic shutdown, fake node, and fake or mock SDK support.
4. Add launch scripts with a side-effect-free `--help` path and clear diagnostics for missing SDKs or devices.
5. Add deploy, collection, evaluation, open-loop, or RL configs only where the robot supports those workflows.
6. Add startup, fake-node, serialization, config, visualization, environment-isolation, and adapter contract tests.

Use existing shared transport and fake-hardware helpers. Do not add compatibility paths for obsolete names unless the user explicitly requests them.

## Validate In Stages

Run static checks, import checks, fake-node tests, config loading, serialization tests, and visualization checks first. State clearly that these are offline evidence.

Before opening a live camera, identify the exact device, serial, physical position, stream, resolution, frame rate, duration, and whether any setting may change. Ask for explicit confirmation, then use `$calibrate-cameras` for exposure or white-balance work.

Before any live SDK connection, motor enable, gripper command, or arm motion, hand off to `$debug-robot-sdk`. Do not execute a live probe directly from this skill. A confirmation for one device or stage does not authorize another device or stage.

## Report Completion

Report the adapter paths, SDK/URDF provenance, left/right mappings, isolated environment, configs, tests, and each evidence level as static, unit, fake, read-only live, or mutating live. List every skipped live check and the exact confirmation still required.
