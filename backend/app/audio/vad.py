"""
Real-Time Voice Activity Detection (VAD) Engine.
Integrates quantized Silero VAD v6 with stateful recurrent streaming,
hangover timing, and rolling pre-roll buffer to prevent audio clipping.
"""

import os
import time
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np

from app.audio.buffer import RollingPreRollBuffer


class SpeechEventType(str, Enum):
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    HANGOVER = "hangover"
    UTTERANCE_END = "utterance_end"


@dataclass
class SpeechSegment:
    """Represents an isolated speech utterance ready for ASR and translation."""
    audio: np.ndarray
    sample_rate: int
    duration_sec: float
    start_time: float
    end_time: float
    rms_dbfs: float
    peak_amp: float
    speech_probs: List[float] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.audio)


def calculate_rms_dbfs(audio: np.ndarray) -> float:
    """Calculate Root Mean Square energy in decibels relative to full scale (dBFS)."""
    if len(audio) == 0:
        return -100.0
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    # 20 * log10(rms)
    db = 20.0 * np.log10(max(rms, 1e-6))
    return float(np.clip(db, -100.0, 0.0))


class StreamingSileroVAD:
    """
    Streaming Voice Activity Detector using stateful Silero VAD ONNX.
    Operates on 512-sample frames (32ms at 16kHz) with recurrent hidden state preservation.
    """

    FRAME_SIZE: int = 512
    CONTEXT_SIZE: int = 64
    SAMPLE_RATE: int = 16000

    def __init__(
        self,
        start_threshold: float = 0.35,
        end_threshold: float = 0.25,
        hangover_ms: int = 450,
        min_speech_ms: int = 250,
        max_speech_ms: int = 8000,
        preroll_ms: int = 350,
        model_path: Optional[str] = None
    ):
        self.start_threshold = float(start_threshold)
        self.end_threshold = float(end_threshold)
        self.hangover_ms = int(hangover_ms)
        self.min_speech_ms = int(min_speech_ms)
        self.max_speech_ms = int(max_speech_ms)

        # Internal state machine
        self._state = SpeechEventType.SILENCE
        self._hangover_time_ms = 0.0
        self._active_speech_frames: List[np.ndarray] = []
        self._speech_probs: List[float] = []
        self._utterance_start_time: float = 0.0

        # Rolling pre-roll buffer to prevent cutting off early consonants
        self._preroll = RollingPreRollBuffer(sample_rate=self.SAMPLE_RATE, duration_ms=preroll_ms)

        # Recurrent ONNX states
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT_SIZE, dtype=np.float32)

        # Load ONNX session
        self._session = self._init_onnx_session(model_path)

    def _init_onnx_session(self, model_path: Optional[str]):
        import onnxruntime
        if model_path is None or not os.path.exists(model_path):
            import faster_whisper.vad as f_vad
            assets_path = f_vad.get_assets_path()
            model_path = os.path.join(assets_path, "silero_vad_v6.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD model not found at: {model_path}")

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        opts.enable_cpu_mem_arena = False

        return onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts
        )

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[float, SpeechEventType, Optional[SpeechSegment]]:
        """
        Process a single 512-sample (32ms) frame at 16kHz.
        Returns:
            (speech_probability, current_event_type, completed_speech_segment_if_any)
        """
        ts = time.time() if timestamp is None else timestamp
        flat_frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if len(flat_frame) != self.FRAME_SIZE:
            raise ValueError(f"Expected frame size {self.FRAME_SIZE}, got {len(flat_frame)}")

        # Construct 576-sample tensor [context (64) + current_frame (512)]
        x = np.concatenate([self._context, flat_frame]).reshape(1, self.FRAME_SIZE + self.CONTEXT_SIZE)
        self._context = flat_frame[-self.CONTEXT_SIZE:].copy()

        # Run ONNX inference
        out, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        prob = float(out[0])
        frame_ms = (self.FRAME_SIZE / self.SAMPLE_RATE) * 1000.0

        event = SpeechEventType.SILENCE
        completed_segment: Optional[SpeechSegment] = None

        if self._state == SpeechEventType.SILENCE:
            if prob >= self.start_threshold:
                # Transition: SILENCE -> SPEECH
                self._state = SpeechEventType.SPEECH
                event = SpeechEventType.SPEECH_START
                self._utterance_start_time = ts
                self._active_speech_frames = []
                self._speech_probs = [prob]

                # Prepend the pre-roll buffer so we don't clip the first syllable
                preroll_audio = self._preroll.get_preroll()
                if len(preroll_audio) > 0:
                    self._active_speech_frames.append(preroll_audio)

                self._active_speech_frames.append(flat_frame.copy())
            else:
                # Keep rolling pre-roll buffer fresh
                self._preroll.push(flat_frame)
                event = SpeechEventType.SILENCE

        elif self._state == SpeechEventType.SPEECH:
            self._active_speech_frames.append(flat_frame.copy())
            self._speech_probs.append(prob)

            current_duration_ms = (sum(len(f) for f in self._active_speech_frames) / self.SAMPLE_RATE) * 1000.0

            # Force emission if speaker has been talking continuously for max_speech_ms
            if current_duration_ms >= self.max_speech_ms:
                completed_segment = self._build_segment(end_time=ts)
                self._active_speech_frames = []
                self._speech_probs = []
                self._state = SpeechEventType.SPEECH
                self._utterance_start_time = ts
                event = SpeechEventType.UTTERANCE_END
            elif prob < self.end_threshold:
                # Speech energy dipped; enter hangover
                self._state = SpeechEventType.HANGOVER
                self._hangover_time_ms = frame_ms
                event = SpeechEventType.HANGOVER
            else:
                event = SpeechEventType.SPEECH

        elif self._state == SpeechEventType.HANGOVER:
            self._active_speech_frames.append(flat_frame.copy())
            self._speech_probs.append(prob)

            if prob >= self.start_threshold:
                # Speaker paused briefly and resumed speaking!
                self._state = SpeechEventType.SPEECH
                self._hangover_time_ms = 0.0
                event = SpeechEventType.SPEECH
            else:
                self._hangover_time_ms += frame_ms
                if self._hangover_time_ms >= self.hangover_ms:
                    # Hangover period expired; user finished speaking
                    completed_segment = self._build_segment(end_time=ts)
                    self._active_speech_frames = []
                    self._speech_probs = []
                    self._hangover_time_ms = 0.0
                    self._state = SpeechEventType.SILENCE
                    self._preroll.clear()
                    event = SpeechEventType.UTTERANCE_END
                else:
                    event = SpeechEventType.HANGOVER

        return prob, event, completed_segment

    def _build_segment(self, end_time: float) -> Optional[SpeechSegment]:
        """Aggregate frames and validate against minimum speech duration."""
        if not self._active_speech_frames:
            return None

        audio = np.concatenate(self._active_speech_frames, axis=0)
        duration_sec = len(audio) / self.SAMPLE_RATE
        duration_ms = duration_sec * 1000.0

        if duration_ms < self.min_speech_ms:
            # Too short; likely a click, cough, or microphone bump
            return None

        rms_db = calculate_rms_dbfs(audio)
        peak_amp = float(np.max(np.abs(audio)))

        return SpeechSegment(
            audio=audio,
            sample_rate=self.SAMPLE_RATE,
            duration_sec=duration_sec,
            start_time=self._utterance_start_time,
            end_time=end_time,
            rms_dbfs=rms_db,
            peak_amp=peak_amp,
            speech_probs=list(self._speech_probs)
        )

    def reset(self) -> None:
        """Reset internal recurrent states and clear audio buffers."""
        self._state = SpeechEventType.SILENCE
        self._hangover_time_ms = 0.0
        self._active_speech_frames.clear()
        self._speech_probs.clear()
        self._preroll.clear()
        self._h.fill(0)
        self._c.fill(0)
        self._context.fill(0)
