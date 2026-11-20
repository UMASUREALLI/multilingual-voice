"""
Data Models for Speaker Consent Verification and Voice Cloning Profiles.
Zero disk persistence: voice profiles are ephemeral and destroyed upon session termination.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import time


class ConsentStatus(str, Enum):
    """Status of ethical voice cloning consent."""
    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    REJECTED = "rejected"


# Required consent phrases across languages
MANDATORY_CONSENT_PHRASES: Dict[str, List[str]] = {
    "en": [
        "i authorize this ai system to clone and use my voice for real-time translation",
        "i consent to having my voice cloned for this call translation",
        "i authorize the use of my voice for translation"
    ],
    "te": [
        "నా స్వరాన్ని అనువాదం కోసం ఉపయోగించడానికి నేను అనుమతిస్తున్నాను",
        "రియల్ టైమ్ అనువాదం కోసం నా వాయిస్ క్లోనింగ్ కి నేను సమ్మతిస్తున్నాను"
    ],
    "hi": [
        "मैं इस अनुवाद के लिए अपनी आवाज का उपयोग करने की अनुमति देता हूं",
        "मैं रियल टाइम अनुवाद के लिए अपनी आवाज क्लोन करने की सहमति देता हूं"
    ],
    "ta": [
        "நிகழ்நேர மொழிபெயர்ப்பிற்காக எனது குரலைப் பயன்படுத்த நான் அங்கீகரிக்கிறேன்"
    ]
}


@dataclass
class AcousticFeatures:
    """Acoustic characteristics extracted from raw speaker audio in memory."""
    f0_median_hz: float          # Median fundamental frequency
    f0_std_hz: float             # Pitch variation
    estimated_gender: str        # "male" | "female"
    speaking_rate_wpm: float     # Estimated words per minute
    energy_rms_dbfs: float       # Average signal energy
    spectral_centroid_hz: float  # Timbre brightness
    audio_duration_sec: float    # Sample length used for profiling

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f0_median_hz": round(self.f0_median_hz, 1),
            "f0_std_hz": round(self.f0_std_hz, 1),
            "estimated_gender": self.estimated_gender,
            "speaking_rate_wpm": round(self.speaking_rate_wpm, 1),
            "energy_rms_dbfs": round(self.energy_rms_dbfs, 1),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 1),
            "audio_duration_sec": round(self.audio_duration_sec, 2),
        }


@dataclass
class ConsentVerificationResult:
    """Outcome of consent phrase verification."""
    is_valid: bool
    status: ConsentStatus
    matched_phrase: Optional[str]
    similarity_score: float
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpeakerVoiceProfile:
    """
    Ephemeral in-memory voice profile.
    Destroyed automatically when call disconnects. Zero disk persistence.
    """
    speaker_id: str
    consent_status: ConsentStatus
    acoustic_features: Optional[AcousticFeatures] = None
    assigned_base_voice: Optional[str] = None
    pitch_adjustment: str = "+0Hz"
    rate_adjustment: str = "+0%"
    volume_adjustment: str = "+0%"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "consent_status": self.consent_status.value,
            "acoustic_features": self.acoustic_features.to_dict() if self.acoustic_features else None,
            "assigned_base_voice": self.assigned_base_voice,
            "pitch_adjustment": self.pitch_adjustment,
            "rate_adjustment": self.rate_adjustment,
            "volume_adjustment": self.volume_adjustment,
            "created_at": self.created_at
        }
