"""Fixed-clock alignment for teleop collection raw streams."""

from __future__ import annotations

import bisect
import dataclasses
import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from core.types import CollectionRawBatch, CollectionRawImage, CollectionRawSample, Observation
from robots.base import Robot


@dataclasses.dataclass(frozen=True)
class AlignmentIssue:
    """One QC issue found while aligning raw collection samples."""

    code: str
    detail: str


@dataclasses.dataclass(frozen=True)
class CollectionAlignmentReport:
    """Diagnostics from fixed-grid collection alignment."""

    grid_start: float
    grid_end: float
    image_max_skew: dict[str, float]
    image_stream_stats: dict[str, dict[str, Any]]
    issues: list[AlignmentIssue]


@dataclasses.dataclass(frozen=True)
class _VectorComponent:
    field: str
    key: str
    samples: list[CollectionRawSample]
    group: Any | None
    gripper_samples: list[CollectionRawSample] | None = None
    default_gripper: float | None = None


def align_collection_samples(
    batch: CollectionRawBatch,
    *,
    robot: Robot,
    camera_keys: Iterable[str],
    vector_fields: Iterable[str],
    fps: float,
    image_skew_sec: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[list[Observation], CollectionAlignmentReport]:
    """Align raw collection streams to a fixed fps grid.

    Args:
        batch: Raw image/vector samples keyed by camera observation key and vector field.
            Each vector sample is a flat float array, typically [robot.total_action_dim]
            for qpos/action or [8 * n_arms] for eef/action_eef.
        robot: Robot metadata used to hold gripper dimensions instead of interpolating them.
        camera_keys: Required camera observation keys.
        vector_fields: Required vector fields: state_qpos, state_eef, action_qpos, action_eef.
        fps: Target fixed grid rate in Hz.
        image_skew_sec: Maximum accepted nearest-neighbor image skew before QC is marked red.
        start_time: Optional episode lower bound in source-clock seconds. Defaults to
            batch.start_time.
        end_time: Optional episode upper bound in source-clock seconds. Defaults to
            batch.end_time.

    Returns:
        frames: Observations on the fixed grid. Images are nearest-neighbor selected;
            continuous vector dimensions are linearly interpolated; gripper dimensions
            are latest-before held.
        report: Alignment diagnostics including max image skew and QC issues.
    """
    if fps <= 0.0:
        raise ValueError(f"collection alignment fps must be positive, got {fps}")
    lower_bound = batch.start_time if start_time is None else start_time
    upper_bound = batch.end_time if end_time is None else end_time
    required_images = tuple(camera_keys)
    required_vectors = tuple(vector_fields)
    image_series: dict[str, list[CollectionRawSample]] = {}
    vector_series = {
        field: _vector_components(batch, field, robot) for field in required_vectors
    }
    issues: list[AlignmentIssue] = []
    coverage_starts: list[float] = []
    coverage_ends: list[float] = []
    image_stream_stats: dict[str, dict[str, Any]] = {}
    for key in required_images:
        samples = _sorted_samples(batch.images.get(key, []))
        if not samples:
            # Images are optional for alignment: state/action still form a useful
            # episode, and the collection writer inserts black frames for this
            # camera before encoding the videos.  Keep the issue so callers can
            # distinguish a real camera recording from a placeholder one.
            issues.append(
                AlignmentIssue("missing_image_stream", f"{key} has no samples")
            )
            continue
        bounded_samples = _bounded_samples(samples, lower_bound, upper_bound)
        if not bounded_samples:
            issues.append(
                AlignmentIssue(
                    "image_stream_outside_episode",
                    f"{key} has no samples within [{lower_bound}, {upper_bound}]",
                )
            )
            continue
        image_series[key] = bounded_samples
        image_stream_stats[key] = _stream_stats(bounded_samples, image_skew_sec)
        start = bounded_samples[0].timestamp
        end = bounded_samples[-1].timestamp
        coverage_starts.append(start)
        coverage_ends.append(end)
    for components in vector_series.values():
        if not components:
            return _empty_alignment_report(
                AlignmentIssue("missing_vector_field", "required vector field has no streams"),
                image_stream_stats,
            )
        for component in components:
            if not component.samples:
                return _empty_alignment_report(
                    AlignmentIssue(
                        "missing_vector_stream",
                        f"{component.key} has no samples",
                    ),
                    image_stream_stats,
                )
            coverage = _bounded_coverage(component.samples, lower_bound, upper_bound)
            if coverage is None:
                return _empty_alignment_report(
                    AlignmentIssue(
                        "vector_stream_outside_episode",
                        f"{component.key} has no samples within [{lower_bound}, {upper_bound}]",
                    ),
                    image_stream_stats,
                )
            start, end = coverage
            if component.field in {"action_qpos", "action_eef"} and upper_bound is not None:
                end = upper_bound
            coverage_starts.append(start)
            coverage_ends.append(end)
            if component.gripper_samples is not None:
                if not component.gripper_samples:
                    return _empty_alignment_report(
                        AlignmentIssue(
                            "missing_gripper_stream",
                            f"{component.key} gripper has no samples",
                        ),
                        image_stream_stats,
                    )
    if not coverage_starts:
        return _empty_alignment_report(
            AlignmentIssue("missing_required_streams", "no required stream coverage"),
            image_stream_stats,
        )

    grid_start = max(coverage_starts)
    grid_end = min(coverage_ends)
    if grid_end < grid_start:
        return _empty_alignment_report(
            AlignmentIssue(
                "no_common_coverage",
                f"required streams do not overlap: start={grid_start:.6f} end={grid_end:.6f}",
            ),
            image_stream_stats,
            grid_start,
            grid_end,
        )

    interval = 1.0 / fps
    n_frames = int(math.floor((grid_end - grid_start) / interval + 1e-9)) + 1
    tail_tolerance = min(image_skew_sec, 0.5 * interval)
    next_timestamp = grid_start + n_frames * interval
    if next_timestamp <= grid_end + tail_tolerance:
        n_frames += 1
    image_max_skew = {key: 0.0 for key in required_images}
    frames: list[Observation] = []
    discrete_masks: dict[str, np.ndarray | None] = {}

    for frame_index in range(n_frames):
        timestamp = grid_start + frame_index * interval
        images: dict[str, np.ndarray] = {}
        for key, samples in image_series.items():
            sample = _nearest_sample(samples, timestamp)
            skew = abs(sample.timestamp - timestamp)
            image_max_skew[key] = max(image_max_skew[key], skew)
            if skew > image_skew_sec:
                issues.append(
                    AlignmentIssue(
                        code="image_skew_exceeded",
                        detail=(
                            f"frame {frame_index} camera {key} skew {skew:.6f}s "
                            f"exceeds {image_skew_sec:.6f}s"
                        ),
                    )
                )
            images[key] = _image_sample_value(sample)

        vectors: dict[str, np.ndarray] = {}
        for field, components in vector_series.items():
            parts: list[np.ndarray] = []
            for component in components:
                if component.key not in discrete_masks:
                    discrete_masks[component.key] = _discrete_mask(
                        field, component.samples[0].value, robot, component.group
                    )
                parts.append(
                    _interpolate_component(
                        component,
                        timestamp,
                        discrete_masks[component.key],
                    )
                )
            vectors[field] = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

        frames.append(
            Observation(
                timestamp=timestamp,
                images=images,
                state_qpos=vectors.get("state_qpos", np.zeros(0, dtype=np.float32)),
                state_eef=vectors.get("state_eef"),
                action_qpos=vectors.get("action_qpos"),
                action_eef=vectors.get("action_eef"),
            )
        )

    return frames, CollectionAlignmentReport(
        grid_start, grid_end, image_max_skew, image_stream_stats, issues
    )


def _empty_alignment_report(
    issue: AlignmentIssue,
    image_stream_stats: dict[str, dict[str, Any]] | None = None,
    grid_start: float = 0.0,
    grid_end: float = 0.0,
) -> tuple[list[Observation], CollectionAlignmentReport]:
    return [], CollectionAlignmentReport(
        grid_start,
        grid_end,
        {},
        {} if image_stream_stats is None else image_stream_stats,
        [issue],
    )


def _sorted_samples(samples: Iterable[CollectionRawSample]) -> list[CollectionRawSample]:
    return sorted(samples, key=lambda sample: sample.timestamp)


def _image_sample_value(sample: CollectionRawSample) -> np.ndarray:
    value = sample.value
    if isinstance(value, CollectionRawImage):
        return value.decode()
    return np.ascontiguousarray(np.asarray(value))


def _bounded_coverage(
    samples: list[CollectionRawSample],
    lower_bound: float | None,
    upper_bound: float | None,
) -> tuple[float, float] | None:
    start = samples[0].timestamp
    end = samples[-1].timestamp
    if lower_bound is not None:
        start_index = bisect.bisect_left([sample.timestamp for sample in samples], lower_bound)
        if start_index == len(samples):
            return None
        start = samples[start_index].timestamp
    if upper_bound is not None:
        end_index = bisect.bisect_right([sample.timestamp for sample in samples], upper_bound) - 1
        if end_index < 0:
            return None
        end = samples[end_index].timestamp
    if end < start:
        return None
    return start, end


def _bounded_samples(
    samples: list[CollectionRawSample],
    lower_bound: float | None,
    upper_bound: float | None,
) -> list[CollectionRawSample]:
    times = [sample.timestamp for sample in samples]
    start_index = 0 if lower_bound is None else bisect.bisect_left(times, lower_bound)
    end_index = len(samples) if upper_bound is None else bisect.bisect_right(times, upper_bound)
    return samples[start_index:end_index]


def _stream_stats(
    samples: list[CollectionRawSample],
    gap_tolerance_sec: float,
) -> dict[str, Any]:
    count = len(samples)
    start_time = samples[0].timestamp
    end_time = samples[-1].timestamp
    gaps = [
        samples[index].timestamp - samples[index - 1].timestamp
        for index in range(1, count)
    ]
    duration = end_time - start_time
    estimated_hz = None
    if count > 1 and duration > 0.0:
        estimated_hz = (count - 1) / duration
    if not gaps:
        return {
            "sample_count": count,
            "start_time": start_time,
            "end_time": end_time,
            "estimated_hz": estimated_hz,
            "max_gap_sec": 0.0,
            "p99_gap_sec": 0.0,
            "gaps_over_tolerance": 0,
            "max_gap_start_time": None,
            "max_gap_end_time": None,
        }
    max_gap_index = max(range(len(gaps)), key=lambda index: gaps[index])
    max_gap = gaps[max_gap_index]
    return {
        "sample_count": count,
        "start_time": start_time,
        "end_time": end_time,
        "estimated_hz": estimated_hz,
        "max_gap_sec": round(max_gap, 6),
        "p99_gap_sec": round(_percentile(gaps, 0.99), 6),
        "gaps_over_tolerance": sum(gap > gap_tolerance_sec for gap in gaps),
        "max_gap_start_time": samples[max_gap_index].timestamp,
        "max_gap_end_time": samples[max_gap_index + 1].timestamp,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _vector_components(
    batch: CollectionRawBatch,
    field: str,
    robot: Robot,
) -> list[_VectorComponent]:
    full = _sorted_samples(batch.vectors.get(field, []))
    if full:
        return [_VectorComponent(field, field, full, None)]

    groups = robot.arm_groups if field in {"state_eef", "action_eef"} else robot.actuator_groups
    components: list[_VectorComponent] = []
    for group in groups:
        key = f"{field}:{group.name}"
        samples = _sorted_samples(batch.vectors.get(key, []))
        gripper_samples = None
        default_gripper = None
        if field in {"state_qpos", "action_qpos"}:
            samples, gripper_samples = _split_group_gripper_samples(
                batch, field, group, samples
            )
            if group.gripper_index is not None:
                default_gripper = _initial_gripper_value(robot, group)
        components.append(
            _VectorComponent(field, key, samples, group, gripper_samples, default_gripper)
        )
    return components


def _nearest_sample(samples: list[CollectionRawSample], timestamp: float) -> CollectionRawSample:
    times = [sample.timestamp for sample in samples]
    index = bisect.bisect_left(times, timestamp)
    if index == 0:
        return samples[0]
    if index == len(samples):
        return samples[-1]
    before = samples[index - 1]
    after = samples[index]
    if abs(timestamp - before.timestamp) <= abs(after.timestamp - timestamp):
        return before
    return after


def _interpolate_component(
    component: _VectorComponent,
    timestamp: float,
    discrete_mask: np.ndarray | None,
) -> np.ndarray:
    if (
        component.group is not None
        and component.group.gripper_index is not None
        and (component.gripper_samples is not None or component.default_gripper is not None)
    ):
        arm = _interpolate_samples(component.samples, timestamp, None)
        if component.gripper_samples is None:
            gripper_value = component.default_gripper
            assert gripper_value is not None
        elif (
            component.default_gripper is not None
            and timestamp < component.gripper_samples[0].timestamp
        ):
            gripper_value = component.default_gripper
        else:
            gripper = _latest_discrete_sample(component.gripper_samples, timestamp)
            value = np.asarray(gripper.value, dtype=np.float32).reshape(-1)
            gripper_value = float(value[0])
        return np.insert(arm, component.group.gripper_index, gripper_value).astype(np.float32)
    return _interpolate_samples(component.samples, timestamp, discrete_mask)


def _split_group_gripper_samples(
    batch: CollectionRawBatch,
    field: str,
    group: Any,
    samples: list[CollectionRawSample],
) -> tuple[list[CollectionRawSample], list[CollectionRawSample] | None]:
    if group.gripper_index is None:
        return samples, None

    arm_samples: list[CollectionRawSample] = []
    gripper_samples: list[CollectionRawSample] = []
    for sample in samples:
        value = np.asarray(sample.value, dtype=np.float32)
        if value.shape == (group.dof,):
            gripper_samples.append(
                CollectionRawSample(
                    timestamp=sample.timestamp,
                    value=np.asarray([value[group.gripper_index]], dtype=np.float32),
                )
            )
            arm_samples.append(
                CollectionRawSample(
                    timestamp=sample.timestamp,
                    value=np.delete(value, group.gripper_index).astype(np.float32),
                )
            )
            continue
        arm_samples.append(sample)

    for key in (f"gripper:{field}:{group.name}", f"{field}:gripper:{group.name}"):
        for sample in batch.vectors.get(key, []):
            value = np.asarray(sample.value, dtype=np.float32).reshape(-1)
            if value.shape[0] == 0:
                continue
            gripper_samples.append(
                CollectionRawSample(
                    timestamp=sample.timestamp,
                    value=np.asarray([value[0]], dtype=np.float32),
                )
            )

    if not gripper_samples:
        return arm_samples, None
    return arm_samples, _sorted_samples(gripper_samples)


def _latest_discrete_sample(
    samples: list[CollectionRawSample], timestamp: float
) -> CollectionRawSample:
    times = [sample.timestamp for sample in samples]
    index = bisect.bisect_right(times, timestamp)
    if index == 0:
        return samples[0]
    return samples[index - 1]


def _initial_gripper_value(robot: Robot, group: Any) -> float:
    offset = 0
    for candidate in robot.actuator_groups:
        if candidate.name == group.name:
            assert group.gripper_index is not None
            return float(robot.initial_qpos[offset + group.gripper_index])
        offset += candidate.dof
    raise ValueError(f"unknown actuator group {group.name!r}")


def _interpolate_samples(
    samples: list[CollectionRawSample],
    timestamp: float,
    discrete_mask: np.ndarray | None,
) -> np.ndarray:
    times = [sample.timestamp for sample in samples]
    index = bisect.bisect_left(times, timestamp)
    if index == 0:
        return np.asarray(samples[0].value, dtype=np.float32).copy()
    if index < len(samples) and times[index] == timestamp:
        return np.asarray(samples[index].value, dtype=np.float32).copy()
    if index == len(samples):
        return np.asarray(samples[-1].value, dtype=np.float32).copy()
    before = samples[index - 1]
    after = samples[index]
    left = np.asarray(before.value, dtype=np.float32)
    right = np.asarray(after.value, dtype=np.float32)
    duration = after.timestamp - before.timestamp
    if duration <= 0.0:
        out = right.copy()
    else:
        alpha = float((timestamp - before.timestamp) / duration)
        out = (left + (right - left) * alpha).astype(np.float32)
    if discrete_mask is not None and discrete_mask.shape == out.shape:
        out[discrete_mask] = left[discrete_mask]
    return out


def _discrete_mask(
    field: str,
    value: Any,
    robot: Robot,
    group: Any | None = None,
) -> np.ndarray | None:
    arr = np.asarray(value)
    if field in {"state_qpos", "action_qpos"} and group is not None:
        mask = np.zeros(arr.shape[0], dtype=bool)
        if group.gripper_index is not None and group.gripper_index < arr.shape[0]:
            mask[group.gripper_index] = True
        return mask
    if field in {"state_qpos", "action_qpos"} and arr.shape == (robot.total_action_dim,):
        return np.asarray(robot.gripper_mask, dtype=bool)
    if field in {"state_eef", "action_eef"} and arr.ndim == 1 and arr.shape[0] % 8 == 0:
        mask = np.zeros(arr.shape[0], dtype=bool)
        mask[7::8] = True
        return mask
    return None
