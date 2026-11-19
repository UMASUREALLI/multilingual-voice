"""
Continuous Live Microphone Audio Capture Streamer.
Streams 16kHz mono audio directly into memory using non-blocking hardware callbacks.
Zero disk writes. Integrates real-time Silero VAD and rolling circular ring buffers.
"""

import sys
import time
import queue
import threading
from typing import Optional, Callable, List, Dict, Any
import numpy as np
import sounddevice as sd

from app.audio.vad import StreamingSileroVAD, SpeechEventType, SpeechSegment, calculate_rms_dbfs
from app.audio.buffer import AudioRingBuffer


class LiveAudioStreamer:
    """
    Asynchronous microphone capture engine.
    Runs sounddevice.InputStream in a non-blocking C-callback, streams raw 16kHz
    PCM chunks to an internal processing queue, and triggers event callbacks.
    """

    SAMPLE_RATE: int = 16000
    BLOCK_SIZE: int = 512  # 32ms at 16kHz
    CHANNELS: int = 1

    def __init__(
        self,
        device_index: Optional[int] = None,
        vad_start_threshold: float = 0.45,
        vad_end_threshold: float = 0.30,
        hangover_ms: int = 450,
        min_speech_ms: int = 250,
        max_speech_ms: int = 8000,
        preroll_ms: int = 350,
        buffer_capacity_sec: float = 30.0
    ):
        self.device_index = device_index
        self._running = False
        self._paused = False
        self._muted = False
        self._ducking_factor = 1.0  # 1.0 = normal, 0.0 = completely ducked

        # In-memory circular audio history (last 30 seconds for debugging/rewind)
        capacity_samples = int(self.SAMPLE_RATE * buffer_capacity_sec)
        self.ring_buffer = AudioRingBuffer(capacity_samples=capacity_samples)

        # Stateful VAD
        self.vad = StreamingSileroVAD(
            start_threshold=vad_start_threshold,
            end_threshold=vad_end_threshold,
            hangover_ms=hangover_ms,
            min_speech_ms=min_speech_ms,
            max_speech_ms=max_speech_ms,
            preroll_ms=preroll_ms
        )

        # Threading & queues
        self._raw_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._worker_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None

        # Callbacks
        self._on_frame_callbacks: List[Callable[[np.ndarray, float, float, SpeechEventType], None]] = []
        self._on_speech_start_callbacks: List[Callable[[float], None]] = []
        self._on_utterance_complete_callbacks: List[Callable[[SpeechSegment], None]] = []

        # Current live metrics
        self.current_rms_dbfs: float = -100.0
        self.current_speech_prob: float = 0.0
        self.current_event: SpeechEventType = SpeechEventType.SILENCE

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """List all valid audio input devices detected on the host system."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name"),
                        "channels": dev.get("max_input_channels"),
                        "default_samplerate": dev.get("default_samplerate"),
                        "hostapi": dev.get("hostapi")
                    })
        except Exception as e:
            print(f"Error querying audio devices: {e}", file=sys.stderr)
        return devices

    def register_on_frame(self, cb: Callable[[np.ndarray, float, float, SpeechEventType], None]) -> None:
        """Register callback for every 32ms audio frame: cb(audio_frame, rms_dbfs, speech_prob, event_type)."""
        self._on_frame_callbacks.append(cb)

    def register_on_speech_start(self, cb: Callable[[float], None]) -> None:
        """Register callback triggered when speech onset is detected: cb(start_timestamp)."""
        self._on_speech_start_callbacks.append(cb)

    def register_on_utterance_complete(self, cb: Callable[[SpeechSegment], None]) -> None:
        """Register callback triggered when a speech utterance completes: cb(speech_segment)."""
        self._on_utterance_complete_callbacks.append(cb)

    def set_muted(self, muted: bool) -> None:
        """Mute/unmute microphone input (e.g., during active synthetic voice playback)."""
        self._muted = bool(muted)

    def set_ducking(self, factor: float) -> None:
        """Set ducking attenuation [0.0 - 1.0] for software echo control."""
        self._ducking_factor = float(np.clip(factor, 0.0, 1.0))

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """
        Real-time PortAudio C-callback. Must remain non-blocking.
        Copies data immediately into the processing queue.
        """
        if status:
            # Overflow or underflow warning
            pass

        if not self._running or self._paused:
            return

        frame = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()

        if self._muted:
            frame.fill(0)
        elif self._ducking_factor < 1.0:
            frame = frame * self._ducking_factor

        try:
            self._raw_queue.put_nowait((frame, time.time()))
        except queue.Full:
            # If queue overflows, drop oldest frame to preserve real-time low latency
            try:
                self._raw_queue.get_nowait()
                self._raw_queue.put_nowait((frame, time.time()))
            except (queue.Empty, queue.Full):
                pass

    def _processing_loop(self) -> None:
        """Background thread that consumes raw frames, runs VAD, and dispatches callbacks."""
        while self._running:
            try:
                frame, timestamp = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Update historical ring buffer
            self.ring_buffer.write(frame)

            # Compute audio metrics
            rms_db = calculate_rms_dbfs(frame)
            self.current_rms_dbfs = rms_db

            # Run stateful Silero VAD
            try:
                prob, event, segment = self.vad.process_frame(frame, timestamp=timestamp)
                self.current_speech_prob = prob
                self.current_event = event
            except Exception as e:
                print(f"VAD error: {e}", file=sys.stderr)
                continue

            # Fire callbacks
            for cb in self._on_frame_callbacks:
                try:
                    cb(frame, rms_db, prob, event)
                except Exception as ex:
                    print(f"Error in on_frame callback: {ex}", file=sys.stderr)

            if event == SpeechEventType.SPEECH_START:
                for cb in self._on_speech_start_callbacks:
                    try:
                        cb(timestamp)
                    except Exception as ex:
                        print(f"Error in on_speech_start callback: {ex}", file=sys.stderr)

            if segment is not None:
                for cb in self._on_utterance_complete_callbacks:
                    try:
                        cb(segment)
                    except Exception as ex:
                        print(f"Error in on_utterance_complete callback: {ex}", file=sys.stderr)

    def start(self) -> None:
        """Start microphone audio capture stream and background processing thread."""
        if self._running:
            return

        self._running = True
        self._paused = False
        self.vad.reset()

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

        # Start PortAudio hardware InputStream
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCK_SIZE,
            device=self.device_index,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._audio_callback
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop microphone capture and terminate worker thread cleanly."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        self.vad.reset()

    def pause(self) -> None:
        """Pause audio capture without destroying the stream."""
        self._paused = True

    def resume(self) -> None:
        """Resume audio capture."""
        self._paused = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
