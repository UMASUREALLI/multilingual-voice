"""
Audio Playback Engine with Acoustic Echo Feedback Prevention.
Manages real-time audio playback through speakers/headphones using sounddevice.
Integrates hardware-level ducking/muting callbacks with LiveAudioStreamer to prevent
acoustic echo feedback loops during bidirectional calls.
Supports barge-in interruption and sequential utterance queuing.
"""

import time
import queue
import threading
import logging
from typing import Optional, Callable, List, Tuple
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioPlaybackEngine:
    """
    Thread-safe real-time audio output engine.
    Controls playback queue, prevents echo loops by coordinating with input streamers,
    and supports instant barge-in cancellation.
    """

    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        self._is_playing = threading.Event()
        self._stop_requested = threading.Event()
        self._play_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # State callbacks for Acoustic Echo Feedback Prevention
        self._on_playback_start_callbacks: List[Callable[[], None]] = []
        self._on_playback_end_callbacks: List[Callable[[], None]] = []

        # Current playback stream
        self._active_stream: Optional[sd.OutputStream] = None

    @property
    def is_playing(self) -> bool:
        """Returns True if synthetic speech is actively playing through the speakers."""
        return self._is_playing.is_set()

    def register_on_playback_start(self, cb: Callable[[], None]) -> None:
        """Register callback triggered when playback starts (e.g. mute/duck microphone)."""
        self._on_playback_start_callbacks.append(cb)

    def register_on_playback_end(self, cb: Callable[[], None]) -> None:
        """Register callback triggered when playback completes (e.g. unmute/unduck microphone)."""
        self._on_playback_end_callbacks.append(cb)

    def _notify_playback_start(self) -> None:
        self._is_playing.set()
        for cb in self._on_playback_start_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Error in on_playback_start callback: {e}")

    def _notify_playback_end(self) -> None:
        self._is_playing.clear()
        for cb in self._on_playback_end_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Error in on_playback_end callback: {e}")

    def start(self) -> None:
        """Start the background playback queue worker."""
        if self._running:
            return
        self._running = True
        self._stop_requested.clear()
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True, name="AudioPlaybackWorker")
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop worker and abort all active playback."""
        self._running = False
        self.interrupt()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def interrupt(self) -> None:
        """
        Barge-in interrupt: Immediately abort current playback and clear pending queue.
        Used when the user interrupts the AI or remote party starts speaking.
        """
        self._stop_requested.set()
        # Clear queue
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
                self._play_queue.task_done()
            except queue.Empty:
                break
        sd.stop()
        self._notify_playback_end()

    def play(self, audio: np.ndarray, sample_rate: int = 24000, blocking: bool = False) -> None:
        """
        Play audio array. If blocking=True, plays synchronously and returns when finished.
        If blocking=False, enqueues to the background playback worker.
        """
        if blocking:
            self._play_synchronous(audio, sample_rate)
        else:
            if not self._running:
                self.start()
            self._play_queue.put((audio, sample_rate))

    def _play_synchronous(self, audio: np.ndarray, sample_rate: int) -> None:
        """Synchronous audio playback with echo loop prevention notifications."""
        self._notify_playback_start()
        try:
            sd.play(audio, samplerate=sample_rate, device=self.device_index, blocking=True)
        finally:
            self._notify_playback_end()

    def _queue_worker(self) -> None:
        """Continuous consumer thread that drains the playback queue."""
        while self._running:
            try:
                item = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio, sample_rate = item
            self._stop_requested.clear()
            self._notify_playback_start()

            try:
                sd.play(audio, samplerate=sample_rate, device=self.device_index)
                duration = len(audio) / sample_rate
                start_t = time.time()

                # Poll in 50ms intervals so interrupt() can abort playback immediately
                while time.time() - start_t < duration:
                    if self._stop_requested.is_set() or not self._running:
                        sd.stop()
                        break
                    time.sleep(0.05)
                sd.stop()
            except Exception as e:
                logger.error(f"Playback error: {e}")
            finally:
                self._notify_playback_end()
                self._play_queue.task_done()


def link_playback_to_streamer(player: AudioPlaybackEngine, streamer: "LiveAudioStreamer", duck_only: bool = False) -> None:
    """
    Convenience helper to link an AudioPlaybackEngine to a LiveAudioStreamer.
    Automatically prevents acoustic feedback loops:
      - If duck_only is False: Completely mutes mic during synthetic voice playback.
      - If duck_only is True: Ducks mic volume to 10% (-20dB) during playback.
    """
    if duck_only:
        player.register_on_playback_start(lambda: streamer.set_ducking(0.10))
        player.register_on_playback_end(lambda: streamer.set_ducking(1.0))
    else:
        player.register_on_playback_start(lambda: streamer.set_muted(True))
        player.register_on_playback_end(lambda: streamer.set_muted(False))
