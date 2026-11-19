"""
Data models for Text-to-Speech (TTS) Synthesis and Voice Profiles.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VoiceProfile:
    """Metadata describing a neural TTS voice."""
    voice_id: str
    name: str
    gender: str          # "male" | "female"
    language_code: str   # "te", "hi", "en", "ta", etc.
    locale: str          # "te-IN", "hi-IN", "en-US", etc.
    description: str = ""


@dataclass
class SpeechSynthesisRequest:
    """Input payload for text-to-speech synthesis."""
    text: str
    language_code: str
    gender: str = "female"
    voice_id: Optional[str] = None
    rate: str = "+0%"     # e.g. "+10%", "-5%"
    pitch: str = "+0Hz"   # e.g. "+20Hz", "-15Hz"
    volume: str = "+0%"


@dataclass
class SpeechSynthesisResult:
    """Output payload from text-to-speech synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration_sec: float
    latency_ms: float
    voice_id: str
    language_code: str
    num_chunks: int = 1

    @property
    def num_samples(self) -> int:
        return len(self.audio)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 2),
            "latency_ms": round(self.latency_ms, 1),
            "voice_id": self.voice_id,
            "language_code": self.language_code,
            "num_chunks": self.num_chunks,
            "num_samples": self.num_samples
        }
