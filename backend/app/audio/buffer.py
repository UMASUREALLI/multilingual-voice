"""
Thread-safe circular in-memory audio buffers for real-time speech streaming.
Avoids disk writes and prevents clipping of early consonant onsets.
"""

import threading
import numpy as np
from typing import Optional


class AudioRingBuffer:
    """
    Fixed-capacity circular ring buffer for raw PCM/float32 audio frames.
    Optimized for continuous real-time audio capture with zero allocations in the hot path.
    """

    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32):
        self.capacity = int(capacity_samples)
        self.dtype = dtype
        self._buffer = np.zeros(self.capacity, dtype=dtype)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """
        Append audio samples into the ring buffer.
        Overwrites oldest samples if buffer exceeds capacity.
        """
        if data is None or len(data) == 0:
            return

        flat_data = np.asarray(data, dtype=self.dtype).reshape(-1)
        n = len(flat_data)

        with self._lock:
            if n >= self.capacity:
                # If incoming data exceeds capacity, keep only the latest slice
                self._buffer[:] = flat_data[-self.capacity:]
                self._write_pos = 0
                self._total_written += n
                return

            end_pos = self._write_pos + n
            if end_pos <= self.capacity:
                self._buffer[self._write_pos:end_pos] = flat_data
            else:
                first_part = self.capacity - self._write_pos
                self._buffer[self._write_pos:] = flat_data[:first_part]
                second_part = n - first_part
                self._buffer[:second_part] = flat_data[first_part:]

            self._write_pos = end_pos % self.capacity
            self._total_written += n

    def read_recent(self, n_samples: int) -> np.ndarray:
        """
        Read the most recent n_samples from the buffer in chronological order.
        """
        with self._lock:
            n = min(int(n_samples), min(self._total_written, self.capacity))
            if n <= 0:
                return np.zeros(0, dtype=self.dtype)

            start_pos = (self._write_pos - n) % self.capacity
            if start_pos + n <= self.capacity:
                return self._buffer[start_pos:start_pos + n].copy()
            else:
                first_part = self.capacity - start_pos
                second_part = n - first_part
                out = np.empty(n, dtype=self.dtype)
                out[:first_part] = self._buffer[start_pos:]
                out[first_part:] = self._buffer[:second_part]
                return out

    def clear(self) -> None:
        """Reset the buffer state to zero."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_written = 0

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    @property
    def current_size(self) -> int:
        with self._lock:
            return min(self._total_written, self.capacity)


class RollingPreRollBuffer:
    """
    Maintains a rolling pre-roll window (e.g. 300ms - 500ms) of recent audio frames.
    When VAD detects the start of speech, this pre-roll is prepended to the utterance
    to ensure leading consonants (p, t, k, s) are never clipped.
    """

    def __init__(self, sample_rate: int = 16000, duration_ms: int = 400):
        self.sample_rate = sample_rate
        self.capacity_samples = int(sample_rate * (duration_ms / 1000.0))
        self._ring = AudioRingBuffer(self.capacity_samples, dtype=np.float32)

    def push(self, frame: np.ndarray) -> None:
        """Push a newly captured frame into the rolling pre-roll buffer."""
        self._ring.write(frame)

    def get_preroll(self) -> np.ndarray:
        """Return the captured pre-roll audio frames in chronological order."""
        return self._ring.read_recent(self.capacity_samples)

    def clear(self) -> None:
        """Clear the pre-roll buffer."""
        self._ring.clear()
