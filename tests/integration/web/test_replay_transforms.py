from __future__ import annotations

import json
import struct

import numpy as np

import robots  # noqa: F401
from core.registry import ROBOT_REGISTRY
from robots.utils import UrdfScene


def _decode_blob(blob: bytes) -> tuple[list[str], np.ndarray]:
    assert blob[:8] == b"EVAXFRM1"
    n_frames, n_geoms, hdr_len = struct.unpack("<III", blob[8:20])
    keys = json.loads(blob[20 : 20 + hdr_len].decode("utf-8"))
    floats = np.frombuffer(blob[20 + hdr_len :], dtype="<f4")
    assert floats.size == n_frames * n_geoms * 16
    return keys, floats.reshape(n_frames, n_geoms, 4, 4)


def test_all_transforms_blob_roundtrip_matches_transforms():
    scene = UrdfScene(ROBOT_REGISTRY.build("agilex_piper"))
    seq = np.stack([np.full(14, 0.1 * f, dtype=np.float32) for f in range(4)])

    keys, mats = _decode_blob(scene.all_transforms_blob(seq))

    # Key order must come from transforms() itself so it matches the float layout.
    arms0 = scene.transforms(seq[0])
    expected_keys = [f"{part}/{geom}" for part in arms0 for geom in arms0[part]]
    assert keys == expected_keys
    assert mats.shape[0] == seq.shape[0]
    assert mats.shape[1] == len(expected_keys)

    # Every frame's decoded matrices equal a fresh transforms() call.
    for f in range(seq.shape[0]):
        arms = scene.transforms(seq[f])
        for g, key in enumerate(keys):
            part, geom = key.split("/", 1)
            np.testing.assert_allclose(mats[f, g], np.asarray(arms[part][geom]), rtol=0, atol=1e-5)


def test_arx_r5_dual_arm_scene_keeps_shared_urdf_parts_separate():
    scene = UrdfScene(ROBOT_REGISTRY.build("arx_r5"))

    assert scene.arm_names == ["left_arm", "right_arm"]
    assert [m["name"] for m in scene.static_meshes()] == [
        "base_link.STL",
        "link1.STL",
        "link2.STL",
        "link3.STL",
        "link4.STL",
        "link5.STL",
        "link6.STL",
        "link7.STL",
        "link8.STL",
    ]

    arms = scene.transforms(None)
    assert set(arms) == {"left_arm", "right_arm"}
    assert set(arms["left_arm"]) == set(arms["right_arm"])
    left_base = np.asarray(arms["left_arm"]["base_link.STL"])[:3, 3]
    right_base = np.asarray(arms["right_arm"]["base_link.STL"])[:3, 3]
    np.testing.assert_allclose(left_base, [-0.25, 0.0, 0.0])
    np.testing.assert_allclose(right_base, [0.25, 0.0, 0.0])
