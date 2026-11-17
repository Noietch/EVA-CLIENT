# Vendor-Neutral Camera Control Contract

Adapt vendor SDKs to these conceptual operations when possible:

- `list_devices`: stable serials or identifiers plus model metadata.
- `open_stream`: one exact device, stream, format, resolution, and frame rate.
- `get_capabilities`: supported controls and whether each is readable, writable, volatile, or persistent.
- `get_control`: current value and auto/manual mode.
- `set_control`: one validated value within the reported range and step.
- `capture`: timestamped frames from the confirmed stream.
- `close`: deterministic release even after failure.

Each control should expose its name, current value, range, step, unit, mode, persistence, and interactions with other controls. Preserve vendor-specific controls without pretending unsupported properties are portable.

Current repository examples include RealSense exposure and white-balance controls in `examples/hardware/arx_x5/camera.py`, mixed RealSense and Orbbec capability checks in `examples/hardware/yam`, and per-camera white-balance tooling in `examples/hardware/arx_r5/utils`.
