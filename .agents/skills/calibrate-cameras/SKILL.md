---
name: calibrate-cameras
description: Evaluate and tune exposure and white balance for robot cameras through capability discovery, confirmed live capture, bounded automatic search, or operator-reviewed screenshots. Use after a camera SDK is integrated or when robot camera color, brightness, clipping, flicker, or cross-camera consistency is wrong.
---

# Calibrate Cameras

Use a vendor-neutral, per-camera workflow. Never treat SDK integration as permission to open a camera or change its controls. This skill does not authorize any motor, arm, or gripper action.

## Establish The Camera Contract

For each camera, collect or discover:

- adapter and SDK, stable device serial or identifier, physical mounting position, and EVA observation key;
- resolution, frame rate, pixel format, stream, warm-up time, and intended lighting or scene;
- whether the user wants bounded automatic tuning or screenshot-based operator review;
- a neutral target or known reference when objective white-balance assessment is required;
- whether settings are volatile or persistent and where the final profile belongs.

For left/right or multi-camera systems, ask the user to confirm physical position to serial/index and EVA key before using images. Never rely on enumeration order. Calibrate every device independently; do not copy a left-camera value to the right camera.

## Discover Capabilities Offline

Inspect the adapter and SDK for capability enumeration and control metadata. Determine which controls actually exist, including auto exposure, exposure time, gain, brightness, auto white balance, white-balance temperature, and per-channel gains. Record range, step, unit, current value, auto/manual interaction, persistence, and rollback behavior. Do not assume that all camera families expose the same properties.

## Gate Live Access

Before opening a live stream, tell the user the exact camera, serial/index, physical position, EVA key, stream, resolution, frame rate, capture duration, output location, and that no setting will change yet. Ask for explicit confirmation.

Open only the confirmed camera, warm it up for a bounded interval, capture a baseline, preserve the original settings, and close it unless the next confirmed step needs the stream. If a local screenshot is available, inspect it with the image-viewing tool.

## Evaluate The Baseline

Assess exposure with black and white clipping fractions, luminance percentiles, usable dynamic range, saturation, and stability over several frames. Check blur and flicker because they constrain exposure time and gain. Do not brighten a deliberately dark scene solely to hit an arbitrary mean.

Assess white balance with a neutral target when available. Otherwise use gray-world or channel-ratio signals only when the scene supports that assumption and label the result as subjective. Review channel medians, neutral-region channel ratios, saturation, and visible color cast. A model's visual judgment is advisory, not proof of correct color.

## Tune With Explicit Scope

Before changing settings, show the exact camera, current settings, supported ranges, proposed properties, bounded search range, maximum iterations and duration, persistence behavior, rollback point, and expected effect. Ask for explicit confirmation of that scope.

In automatic mode, change one control by a small supported step, wait for stabilization, capture frames, score them against the baseline, and roll back a worse result. Respect the confirmed iteration and time caps. Do not write persistent settings during search.

In operator-review mode, show the baseline or candidate screenshot, current settings, objective metrics, and the proposed next change. Ask the user whether the image is acceptable or which candidate they prefer before applying another change.

If a proposal exceeds the confirmed bounds, changes another property, targets another camera, or switches between auto and manual control, request new confirmation.

## Persist And Verify

Present the final before/after images, metrics, and exact settings. Ask for explicit confirmation before writing a persistent profile or device setting. Write the per-camera profile atomically, reload it, capture a fresh verification sample, and restore the original settings if persistence or verification fails.

Report each camera's physical mapping, capabilities, original and final controls, images, metrics, subjective judgments, confirmation boundaries, rollback state, and whether verification was live or only offline.
