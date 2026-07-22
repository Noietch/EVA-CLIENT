"""Action buffers for async inference strategies.

- StreamActionBuffer: linear overlap smoothing (from agilex_inference_openpi_rtc.py)
- NaiveActionBuffer: latency-aligned whole-chunk replacement, no blending
- EnsembleActionBuffer: ACT-style temporal ensembling over overlapping chunks
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class StreamActionBuffer:
    """Chunk buffer with latency trim and linear overlap smoothing.

    New chunks are integrated via integrate_new_chunk():
    1. Trim front by min(k, max_k) steps for latency compensation.
    2. Smooth overlap region: linearly blend old (100%->0%) with new (0%->100%).
    3. Append non-overlapping tail from the new chunk.
    4. Reset k counter to 0.

    Each pop_next_action() increments k (steps consumed since last chunk).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cur_chunk: deque[np.ndarray] = deque()
        self._k: int = 0
        self._last_action: np.ndarray | None = None
        self._revision: int = 0

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        max_k: int = 0,
        min_m: int = 8,
    ) -> None:
        """Integrate a new inference chunk with latency trim and linear overlap smoothing."""
        actions = np.asarray(actions_chunk, dtype=np.float32)
        if len(actions) == 0:
            return
        max_k = max(0, int(max_k))
        min_m = max(1, int(min_m))

        while True:
            with self._lock:
                revision = self._revision
                consumed = self._k
                old_list = list(self._cur_chunk)
                last_action = self._last_action

            drop_n = min(consumed, max_k)
            if drop_n >= len(actions):
                with self._lock:
                    if revision != self._revision:
                        continue
                    return
            new = actions[drop_n:].copy()
            consumed_seed = not old_list and last_action is not None
            if consumed_seed:
                assert last_action is not None
                old = np.repeat(last_action[np.newaxis, :], min_m, axis=0)
            elif old_list:
                old = np.stack(old_list)
                if len(old) < min_m:
                    padding = np.repeat(old[-1][np.newaxis, :], min_m - len(old), axis=0)
                    old = np.concatenate((old, padding), axis=0)
            else:
                combined = new
                prepared_chunk = deque(row.copy() for row in combined)
                with self._lock:
                    if revision != self._revision:
                        continue
                    self._cur_chunk = prepared_chunk
                    self._k = 0
                    self._revision += 1
                    return

            overlap_len = min(len(old), len(new))
            old = old[: len(new)]
            if overlap_len == 1:
                old_weights = np.ones((1, 1), dtype=np.float32)
            else:
                old_weights = np.linspace(
                    1.0, 0.0, overlap_len, dtype=np.float32
                )[:, np.newaxis]
            smoothed = old[:overlap_len] * old_weights + new[:overlap_len] * (
                1.0 - old_weights
            )
            combined = np.concatenate((smoothed, new[overlap_len:]), axis=0)
            prepared_chunk = deque(row.copy() for row in combined)
            with self._lock:
                if revision != self._revision:
                    continue
                self._cur_chunk = prepared_chunk
                self._k = 0
                if consumed_seed:
                    self._last_action = None
                self._revision += 1
                return

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return next action; increment k. Returns None if empty."""
        with self._lock:
            if len(self._cur_chunk) == 0:
                return None
            if len(self._cur_chunk) == 1:
                self._last_action = np.asarray(self._cur_chunk[0], dtype=np.float32).copy()
            act = np.asarray(self._cur_chunk.popleft(), dtype=np.float32)
            self._k += 1
            self._revision += 1
            return act

    def has_actions(self) -> bool:
        """Return True if the current chunk still has actions to pop."""
        with self._lock:
            return len(self._cur_chunk) > 0

    def seed(self, action: np.ndarray) -> None:
        """Anchor the buffer to a known position, clearing all stale state."""
        with self._lock:
            self._cur_chunk.clear()
            self._k = 0
            self._last_action = np.asarray(action, dtype=np.float32).copy()
            self._revision += 1

    def reset(self) -> None:
        """Clear the buffer and all latency/smoothing state."""
        with self._lock:
            self._cur_chunk.clear()
            self._k = 0
            self._last_action = None
            self._revision += 1


class NaiveActionBuffer:
    """Chunk buffer with latency-aligned whole-chunk replacement, no blending.

    On each new chunk, the front is trimmed by min(k, max_k) steps to drop
    actions already consumed since the previous chunk arrived, then the trimmed
    chunk wholly replaces whatever remained in the buffer. There is no overlap
    smoothing — the executed trajectory jumps directly onto the newest chunk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cur_chunk: deque[np.ndarray] = deque()
        self._k: int = 0
        self._last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 0) -> None:
        """Replace the buffer with the new chunk after trimming min(k, max_k) stale front steps."""
        with self._lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            drop_n = min(self._k, max(0, int(max_k)))
            if drop_n >= len(actions_chunk):
                return
            self._cur_chunk = deque(
                np.asarray(a, dtype=np.float32).copy() for a in actions_chunk[drop_n:]
            )
            self._k = 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return next action; increment k. Returns None if empty."""
        with self._lock:
            if len(self._cur_chunk) == 0:
                return None
            if len(self._cur_chunk) == 1:
                self._last_action = np.asarray(self._cur_chunk[0], dtype=np.float32).copy()
            act = np.asarray(self._cur_chunk.popleft(), dtype=np.float32)
            self._k += 1
            return act

    def has_actions(self) -> bool:
        """Return True if the current chunk still has actions to pop."""
        with self._lock:
            return len(self._cur_chunk) > 0

    def seed(self, action: np.ndarray) -> None:
        """Anchor the buffer to a known position, clearing all stale state."""
        with self._lock:
            self._cur_chunk.clear()
            self._k = 0
            self._last_action = np.asarray(action, dtype=np.float32).copy()

    def reset(self) -> None:
        """Clear the buffer and all latency state."""
        with self._lock:
            self._cur_chunk.clear()
            self._k = 0
            self._last_action = None


class EnsembleActionBuffer:
    """ACT-style temporal ensembling over overlapping predicted chunks.

    Reproduces the ACT inference-time aggregation: every chunk produced by the
    background loop is stored keyed by the global step its first action targets.
    To emit the action for the current execution step, all stored chunks whose
    horizon covers that step are gathered and combined with an exponentially
    decaying normalized weighted average, weighting the oldest surviving
    prediction most (weight exp(-m*i), i = age-rank among survivors).
    """

    def __init__(self, exp_weight_m: float = 0.01) -> None:
        self._lock = threading.Lock()
        self._m = max(0.0, float(exp_weight_m))
        # chunks pending integration this step, oldest-first by arrival
        self._chunks: deque[tuple[int, np.ndarray]] = deque()
        self._step: int = 0
        self._last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 0) -> None:
        """Store a new chunk anchored at the current execution step for later ensembling.

        Args:
            actions_chunk: Fresh prediction, shape [T, action_dim].
            max_k: Unused; kept for a uniform buffer interface (latency is handled
                implicitly by the step-anchored overlap).
        """
        _ = max_k
        with self._lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            chunk = np.asarray(actions_chunk, dtype=np.float32).copy()
            self._chunks.append((self._step, chunk))
            self._prune_locked()

    def pop_next_action(self) -> np.ndarray | None:
        """Aggregate all chunks covering the current step, advance, and return the action."""
        with self._lock:
            preds = self._gather_locked(self._step)
            if not preds:
                return None
            n = len(preds)
            if n == 1:
                action = preds[0]
            else:
                w = np.exp(-self._m * np.arange(n, dtype=np.float32))
                w /= w.sum()
                action = np.tensordot(w, np.stack(preds, axis=0), axes=(0, 0)).astype(np.float32)
            self._step += 1
            self._last_action = action.copy()
            self._prune_locked()
            return action

    def has_actions(self) -> bool:
        """Return True if any stored chunk still covers the current step."""
        with self._lock:
            return bool(self._gather_locked(self._step))

    def seed(self, action: np.ndarray) -> None:
        """Anchor the buffer to a known position, clearing all stored chunks."""
        with self._lock:
            self._chunks.clear()
            self._step = 0
            self._last_action = np.asarray(action, dtype=np.float32).copy()

    def reset(self) -> None:
        """Clear all stored chunks and step state."""
        with self._lock:
            self._chunks.clear()
            self._step = 0
            self._last_action = None

    def _gather_locked(self, step: int) -> list[np.ndarray]:
        """Return predictions for ``step`` from every covering chunk, oldest chunk first."""
        preds: list[np.ndarray] = []
        for anchor, chunk in self._chunks:
            offset = step - anchor
            if 0 <= offset < len(chunk):
                preds.append(chunk[offset])
        return preds

    def _prune_locked(self) -> None:
        """Drop chunks whose horizon no longer reaches the current step."""
        while self._chunks:
            anchor, chunk = self._chunks[0]
            if anchor + len(chunk) <= self._step:
                self._chunks.popleft()
            else:
                break
