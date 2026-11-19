"""
Data models for Intelligent Multilingual Translation.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TranslationRequest:
    """Input payload for translation."""
    text: str
    source_language: str
    target_language: str
    context_history: List[str] = field(default_factory=list)
    preserve_entities: bool = True
    speaker_name: Optional[str] = None
    receiver_name: Optional[str] = None


@dataclass
class TranslationResult:
    """Output payload from translation."""
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    source_language_name: str
    target_language_name: str
    backend: str
    latency_ms: float
    preserved_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_language_name": self.source_language_name,
            "target_language_name": self.target_language_name,
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 1),
            "preserved_entities": self.preserved_entities
        }
