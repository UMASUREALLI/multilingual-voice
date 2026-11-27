"""
Data Models for Full-Duplex Bidirectional Call Coordination and Live Subtitles.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import uuid


class CallStatus(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    MUTED = "muted"
    ENDED = "ended"


@dataclass
class CallParticipant:
    """Participant in a bidirectional translated voice call."""
    participant_id: str
    name: str
    native_language: str       # e.g. "te", "en", "hi", "ta"
    target_language: str       # e.g. "en", "te"
    gender: str = "female"
    is_muted: bool = False
    voice_cloned: bool = False
    pitch_mod: str = "+0Hz"
    rate_mod: str = "+0%"


@dataclass
class CallSubtitle:
    """Synchronized dual-language subtitle event for live display."""
    subtitle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    speaker_id: str = ""
    speaker_name: str = ""
    original_language: str = ""
    original_text: str = ""
    target_language: str = ""
    translated_text: str = ""
    asr_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    has_audio: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.subtitle_id,
            "timestamp": round(self.timestamp, 3),
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "original_language": self.original_language,
            "original_text": self.original_text,
            "target_language": self.target_language,
            "translated_text": self.translated_text,
            "asr_latency_ms": round(self.asr_latency_ms, 1),
            "translation_latency_ms": round(self.translation_latency_ms, 1),
            "tts_latency_ms": round(self.tts_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "has_audio": self.has_audio
        }


@dataclass
class CallSession:
    """Full-duplex call session managing two participants."""
    session_id: str
    caller_a: CallParticipant
    caller_b: CallParticipant
    status: CallStatus = CallStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    subtitles: List[CallSubtitle] = field(default_factory=list)

    def get_participant(self, participant_id: str) -> Optional[CallParticipant]:
        if self.caller_a.participant_id == participant_id:
            return self.caller_a
        elif self.caller_b.participant_id == participant_id:
            return self.caller_b
        return None

    def get_recipient(self, speaker_id: str) -> CallParticipant:
        """Get the opposite participant who receives the translated speech."""
        if self.caller_a.participant_id == speaker_id:
            return self.caller_b
        return self.caller_a

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "caller_a": {
                "id": self.caller_a.participant_id,
                "name": self.caller_a.name,
                "native_language": self.caller_a.native_language,
                "target_language": self.caller_a.target_language,
                "voice_cloned": self.caller_a.voice_cloned,
            },
            "caller_b": {
                "id": self.caller_b.participant_id,
                "name": self.caller_b.name,
                "native_language": self.caller_b.native_language,
                "target_language": self.caller_b.target_language,
                "voice_cloned": self.caller_b.voice_cloned,
            },
            "participants": {
                "caller_a": {
                    "id": self.caller_a.participant_id,
                    "name": self.caller_a.name,
                    "language": self.caller_a.native_language,
                    "gender": self.caller_a.gender
                },
                "caller_b": {
                    "id": self.caller_b.participant_id,
                    "name": self.caller_b.name,
                    "language": self.caller_b.native_language,
                    "gender": self.caller_b.gender
                }
            },
            "total_subtitles": len(self.subtitles),
            "created_at": self.created_at
        }
