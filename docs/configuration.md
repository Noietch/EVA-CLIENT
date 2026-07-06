[← Back to README](../README.md)

# ⚙️ Configuration

Configs are `.py` files using mmengine-style `_base_` inheritance with
field-wise recursive deep merge — a preset overrides only what it changes,
`_delete_=True` drops an inherited subtree. `configs/00_base/defaults.py` is
the single source of truth for omitted fields; scalars and whole lists
**replace** the parent value (lists are never concatenated). Deploy presets
inherit a sibling `_base.py`, which in turn inherits `../../00_base/defaults.py`.

```python
# configs/01_deploy/dual_agilex_piper/openpi_qpos.py
_base_ = ['_base.py']                          # → sibling _base.py → ../../00_base/defaults.py

policy = dict(
    type='openpi_rtc',                         # openpi / openpi_rtc / starvla / gr00t / mock / replay
    backend_options=dict(latency_k=4),
)

inference_cfg = dict(
    debug_tasks=['pick up the yellow spoon and place it on the green plate'],
)

# deep merge: override only `args` for the inherited 'async' / 'rtc' strategies
inference_strategies = {
    'async': dict(args=dict(latency_k=4)),
    'rtc':   dict(args=dict(latency_k=4)),
}
```

Startup pipeline: merge the `_base_` chain → build control spaces (joint / EEF)
→ fill derived paths from `work_dir` → validate recording configs → resolve
eval checkpoints (each with its own config, host, port). Failures print a
Python traceback naming the section/file at fault.
