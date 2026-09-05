from __future__ import annotations

import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene

pytestmark = pytest.mark.integration


def test_mesh_payload_preserves_cad_hard_edges_and_source_geometry():
    import trimesh

    scene = UrdfScene(ROBOT_REGISTRY.build("arx_x5"))
    name = scene.static_meshes()[0]["name"]
    urdf_path, raw_name = scene._geom_source[name]
    cube = trimesh.creation.box()
    scene._urdfs[urdf_path].scene.geometry[raw_name] = cube
    payload = scene.mesh_bytes(name)
    vertices, faces = struct.unpack_from("<II", payload, 8)
    positions = np.frombuffer(payload, dtype="<f4", count=vertices * 3, offset=16).reshape(-1, 3)
    normals = np.frombuffer(
        payload, dtype="<f4", count=vertices * 3, offset=16 + vertices * 12
    ).reshape(-1, 3)
    indices = np.frombuffer(payload, dtype="<u4", offset=16 + vertices * 24).reshape(-1, 3)

    assert vertices == 24
    assert faces == 12
    assert len(cube.vertices) == 8
    assert scene.mesh_bytes(name) is payload
    np.testing.assert_allclose([positions.min(axis=0), positions.max(axis=0)], cube.bounds)
    face_normals = trimesh.Trimesh(positions, indices, process=False).face_normals
    np.testing.assert_allclose(normals[indices], np.repeat(face_normals[:, None, :], 3, axis=1))


def test_scene_transforms_serialize_shared_urdf_updates():
    scene = UrdfScene(ROBOT_REGISTRY.build("agilex_piper"))
    active = 0
    max_active = 0
    guard = threading.Lock()
    entered = threading.Event()

    def part_transforms(_part, _qpos_part: np.ndarray, _base: np.ndarray) -> dict:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
            entered.set()
        time.sleep(0.05)
        with guard:
            active -= 1
        return {}

    scene._part_transforms = part_transforms  # type: ignore[method-assign]

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(scene.transforms, np.zeros(14, dtype=np.float32))
        assert entered.wait(timeout=1.0)
        second = executor.submit(scene.transforms, np.ones(14, dtype=np.float32))
        first.result()
        second.result()

    assert max_active == 1
