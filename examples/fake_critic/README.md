# Fake Critic

Run inside the project environment:

```bash
python examples/fake_critic/fake_critic_server.py --port 9100 --delay-ms 8
```

The server sends Critic metadata on connection, then accepts msgpack WebSocket
requests shaped as `{observation, actions}` and returns one scalar `{value}`.
