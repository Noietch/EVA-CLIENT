import pytest
import zmq

import robots  # noqa: F401
from core.config import ConfigDict
from core.registry import ROBOT_REGISTRY
from transport.zmq import WireObservation, _ObservationReader, pack_observation

pytestmark = pytest.mark.integration


def test_new_episode_discards_old_zmq_pipe(tmp_path):
    publisher = zmq.Context.instance().socket(zmq.XPUB)
    publisher.setsockopt(zmq.XPUB_VERBOSE, 1)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = _ObservationReader(
        ConfigDict(
            transport=ConfigDict(
                sub_endpoint=f"tcp://127.0.0.1:{port}", disabled_cameras=[], disabled_groups=[]
            )
        ),
        robot,
        zmq,
        preserve_collection_backlog=True,
    )

    def wait_for_subscription():
        while True:
            assert publisher.poll(2000), "subscriber did not connect"
            if publisher.recv()[0] == 1:
                return

    def publish(timestamp):
        publisher.send(pack_observation(WireObservation(t=timestamp, images={}, state={})))

    try:
        wait_for_subscription()
        for timestamp in range(64):
            publish(float(timestamp))
        assert reader._sub.poll(2000)
        old_socket = reader._sub
        assert reader.clear_collection_backlog() is None
        assert old_socket.closed
        reader.prepare_collection_capture(str(tmp_path))
        wait_for_subscription()
        publish(200.0)
        assert reader._sub.poll(2000)
        snapshot = reader.acquire_collection_raw()
        assert snapshot is not None
        assert snapshot.timestamp == 200.0
        assert reader.acquire_collection_raw() is None
    finally:
        reader.close()
        publisher.close(linger=0)


@pytest.mark.parametrize("ack_matches", [False, True])
def test_action_ack_requires_the_published_timestamp(ack_matches):
    publisher = zmq.Context.instance().socket(zmq.XPUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    robot = ROBOT_REGISTRY.build("agilex_piper")
    reader = _ObservationReader(
        ConfigDict(
            transport=ConfigDict(
                sub_endpoint=f"tcp://127.0.0.1:{port}", disabled_cameras=[], disabled_groups=[]
            )
        ),
        robot,
        zmq,
    )
    try:
        assert publisher.poll(2000), "subscriber did not connect"
        assert publisher.recv()[0] == 1
        publisher.send(
            pack_observation(
                WireObservation(
                    t=1.0,
                    images={},
                    state={},
                    sim_state={
                        "action_time": 123.0 if ack_matches else 122.0,
                        "execution_complete": True,
                    },
                )
            )
        )
        assert reader._sub.poll(2000)
        assert reader.wait_action_applied(123.0, timeout=0.05) is ack_matches
        assert reader.execution_complete()
    finally:
        reader.close()
        publisher.close(linger=0)


def _queued_raw_reader(tmp_path):
    import collections
    import threading
    from types import SimpleNamespace

    class EmptySocket:
        def recv(self, _flags):
            raise zmq.Again()

    reader = _ObservationReader.__new__(_ObservationReader)
    reader._lock = threading.Lock()
    reader._preserve_collection_backlog = True
    reader._capture_latest_only = False
    reader._raw_collection_queue = collections.deque()
    reader._sub = EmptySocket()
    reader._zmq = zmq
    reader._freshness = SimpleNamespace(mark=lambda: None)
    reader._image_mode = "stream"
    reader._collection_journal = None
    reader._collection_journal_dir = tmp_path
    return reader


@pytest.mark.parametrize("latest_only", [False, True])
def test_policy_sampling_reads_latest_while_collection_keeps_fifo(tmp_path, latest_only):
    reader = _queued_raw_reader(tmp_path)
    for index in range(60):
        reader._raw_collection_queue.append(
            pack_observation(WireObservation(t=index / 60, images={}, state={}))
        )
    reader.set_collection_capture_latest_only(latest_only)
    snapshot = reader.acquire_collection_raw()
    assert snapshot.timestamp == pytest.approx(59 / 60 if latest_only else 0)
    assert len(reader._raw_collection_queue) == (0 if latest_only else 59)
    reader.finish_collection_capture()
    assert not reader._capture_latest_only
    assert snapshot.decode_raw() is not None


def test_60hz_camera_sampled_at_30hz_has_no_accumulating_time_lag(tmp_path):
    import numpy as np

    reader = _queued_raw_reader(tmp_path)
    reader.set_collection_capture_latest_only(True)
    times = []
    pixels = []
    try:
        for tick in range(60):
            for index in (2 * tick, 2 * tick + 1):
                reader._raw_collection_queue.append(
                    pack_observation(
                        WireObservation(
                            t=1000 + index / 60,
                            images={"camera": np.full((2, 2, 3), index, dtype=np.uint8)},
                            state={},
                        )
                    )
                )
            snapshot = reader.acquire_collection_raw()
            times.append(snapshot.timestamp)
            raw_image = snapshot.decode_raw().images["camera"][0].value
            assert raw_image._decoded is None
            pixels.append(int(raw_image.decode()[0, 0, 0]))
            raw_image.release_decoded()
        np.testing.assert_allclose(np.diff(times), np.full(59, 1 / 30), atol=1e-10)
        assert times[-1] == pytest.approx(1000 + 119 / 60)
        assert pixels == list(range(1, 120, 2))
        assert not reader._raw_collection_queue
    finally:
        reader.finish_collection_capture()


def test_latest_policy_samples_encode_a_continuous_video_after_capture_finishes(tmp_path):
    import json

    import cv2
    import numpy as np

    from core.recorder.episode import EpisodeLogger

    robot = ROBOT_REGISTRY.build("agilex_piper")
    directory = tmp_path / "recording"
    logger = EpisodeLogger(
        directory,
        robot,
        fps=30,
        dataset_keys=ConfigDict(
            state_key="observations.state.qpos",
            eef_key="observations.state.eef",
            action_key="action",
            video_keys={},
        ),
        async_save=False,
    )
    reader = _queued_raw_reader(tmp_path)
    reader.set_collection_capture_latest_only(True)
    logger.start_episode("synthetic video continuity check")
    for tick in range(60):
        for index in (2 * tick, 2 * tick + 1):
            images = {
                camera.observation_key: np.full((16, 16, 3), index, dtype=np.uint8)
                for camera in robot.observation_schema.cameras
            }
            reader._raw_collection_queue.append(
                pack_observation(
                    WireObservation(
                        t=1000 + index / 60,
                        images=images,
                        state={
                            group.name: qpos
                            for group, qpos in zip(
                                robot.actuator_groups, robot.initial_qpos_by_group(), strict=True
                            )
                        },
                    )
                )
            )
        snapshot = reader.acquire_collection_raw()
        logger.ingest_raw_episode_snapshot(snapshot)
        logger.ingest_policy_action(robot.initial_qpos, robot.initial_qpos, snapshot.timestamp)
    reader.finish_collection_capture()
    assert logger.end_episode()
    logger.finalize()
    metadata = json.loads((directory / "meta/episodes.jsonl").read_text().splitlines()[0])
    assert metadata["length"] == 60
    assert not any(
        issue["code"] == "image_skew_exceeded" for issue in metadata.get("quality_issues", [])
    )
    videos = list(directory.rglob("*.mp4"))
    assert len(videos) == len(robot.observation_schema.cameras)
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 60
        assert cap.get(cv2.CAP_PROP_FPS) == 30
        means = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            means.append(float(frame.mean()))
        cap.release()
        assert len(means) == 60
        assert means[-1] - means[0] > 110
        assert np.max(np.abs(np.diff(means))) < 6
