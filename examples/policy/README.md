# Policy examples

This directory contains lightweight policy API adapters and a local fake policy
for debugging. API adapters forward EVA-CLIENT observations to a provider and
return action chunks; the provider handles training, weight hosting, and inference.

API adapters run as separate services connected to EVA-CLIENT over WebSocket.
Provider SDKs, credentials, and dependencies stay in the example's own environment.
The fake policy uses the existing OpenPI/StarVLA WebSocket protocol and runs in
EVA-CLIENT's environment.

- [DreamLabs / Dream-1](dreamlabs/README.md): OpenPI WebSocket bridge with robot selection.
  Start it with `bash examples/policy/dreamlabs/serve.sh --robot yam`.
- [Fake policy](fake_policy/fake_policy_server.py): synthetic actions for local debugging.
  Start it with `bash examples/policy/fake_policy/run_fake_policy.sh --robot dual_yam`.
