"""Camera streaming shared by hardware nodes with independent process lifetimes."""

import signal
import threading
import time

import msgpack
import numpy as np
import zmq


class CameraPublisher:
    def __init__(self, caches: tuple, endpoint: str, rate: float = 30) -> None:
        self.caches = caches
        self.endpoint = endpoint
        self.interval = 1 / rate
        self.stopped = threading.Event()

    def run(self) -> None:
        publisher = zmq.Context.instance().socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, 1)
        generation = time.monotonic_ns()
        try:
            publisher.bind(self.endpoint)
            signal.signal(signal.SIGINT, lambda *_: self.stopped.set())
            signal.signal(signal.SIGTERM, lambda *_: self.stopped.set())
            while not self.stopped.is_set():
                started = time.monotonic()
                images = {}
                for cache in self.caches:
                    if hasattr(cache, "snapshot_versioned"):
                        versions, frames = cache.snapshot_versioned()
                    else:
                        frames = cache.snapshot()
                        versions = {name: time.monotonic_ns() for name in frames}
                    for name, image in frames.items():
                        version = f"{generation}:{versions[name]}"
                        images[name] = (version, image.shape, image.dtype.str, image.tobytes())
                if not images:
                    self.stopped.wait(self.interval)
                    continue
                publisher.send(msgpack.packb(images, use_bin_type=True))
                self.stopped.wait(max(0, self.interval - (time.monotonic() - started)))
        finally:
            publisher.close(linger=0)
            for cache in self.caches:
                cache.close()


class CameraSource:
    def __init__(self, endpoint: str) -> None:
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(endpoint)
        self.images = {}
        self.versions = {}
        self.received_at = {}

    def snapshot_versioned(self) -> tuple[dict, dict]:
        if self.socket.poll(0):
            payload = msgpack.unpackb(self.socket.recv(), raw=False)
            for name, (version, shape, dtype, data) in payload.items():
                if self.versions.get(name) != version:
                    self.images[name] = np.frombuffer(data, dtype=dtype).reshape(shape)
                    self.received_at[name] = time.monotonic()
                    self.versions[name] = version
        images = {
            name: image
            for name, image in self.images.items()
            if time.monotonic() - self.received_at[name] <= 0.5
        }
        return {name: self.versions[name] for name in images}, images

    def snapshot(self) -> dict:
        return self.snapshot_versioned()[1]

    def hardware_status(self) -> dict:
        return {"camera_stream": "online" if self.snapshot() else "offline"}

    def close(self) -> None:
        self.socket.close(linger=0)
