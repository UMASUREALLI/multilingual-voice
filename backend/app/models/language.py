"""
Language metadata, ISO codes, Unicode script ranges, and LID data structures.
"""

from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageInfo:
    """Metadata for a supported language."""
    code: str
    name: str
    native_name: str
    unicode_range: Optional[Tuple[int, int]] = None
    family: str = "general"


# Comprehensive registry covering major Indian and international languages
SUPPORTED_LANGUAGES: Dict[str, LanguageInfo] = {
    # Major Indian Languages
    "te": LanguageInfo("te", "Telugu", "తెలుగు", (0x0C00, 0x0C7F), "dravidian"),
    "hi": LanguageInfo("hi", "Hindi", "हिन्दी", (0x0900, 0x097F), "indo_aryan"),
    "ta": LanguageInfo("ta", "Tamil", "தமிழ்", (0x0B80, 0x0BFF), "dravidian"),
    "kn": LanguageInfo("kn", "Kannada", "ಕನ್ನಡ", (0x0C80, 0x0CFF), "dravidian"),
    "ml": LanguageInfo("ml", "Malayalam", "മലയാളം", (0x0D00, 0x0D7F), "dravidian"),
    "mr": LanguageInfo("mr", "Marathi", "मराठी", (0x0900, 0x097F), "indo_aryan"),
    "bn": LanguageInfo("bn", "Bengali", "বাংলা", (0x0980, 0x09FF), "indo_aryan"),
    "gu": LanguageInfo("gu", "Gujarati", "ગુજરાતી", (0x0A80, 0x0AFF), "indo_aryan"),
    "pa": LanguageInfo("pa", "Punjabi", "ਪੰਜਾਬੀ", (0x0A00, 0x0A7F), "indo_aryan"),
    "ur": LanguageInfo("ur", "Urdu", "اردو", (0x0600, 0x06FF), "indo_aryan"),
    "or": LanguageInfo("or", "Odia", "ଓଡ଼ିଆ", (0x0B00, 0x0B7F), "indo_aryan"),
    "as": LanguageInfo("as", "Assamese", "অসমীয়া", (0x0980, 0x09FF), "indo_aryan"),
    "ne": LanguageInfo("ne", "Nepali", "नेपाली", (0x0900, 0x097F), "indo_aryan"),

    # Global Languages
    "en": LanguageInfo("en", "English", "English", (0x0041, 0x007A), "germanic"),
    "es": LanguageInfo("es", "Spanish", "Español", None, "romance"),
    "fr": LanguageInfo("fr", "French", "Français", None, "romance"),
    "de": LanguageInfo("de", "German", "Deutsch", None, "germanic"),
    "ja": LanguageInfo("ja", "Japanese", "日本語", (0x3040, 0x30FF), "japonic"),
    "zh": LanguageInfo("zh", "Chinese", "中文", (0x4E00, 0x9FFF), "sino_tibetan"),
    "ar": LanguageInfo("ar", "Arabic", "العربية", (0x0600, 0x06FF), "semitic"),
    "ru": LanguageInfo("ru", "Russian", "Русский", (0x0400, 0x04FF), "slavic"),
    "pt": LanguageInfo("pt", "Portuguese", "Português", None, "romance"),
    "it": LanguageInfo("it", "Italian", "Italiano", None, "romance"),
    "ko": LanguageInfo("ko", "Korean", "한국어", (0xAC00, 0xD7AF), "koreanic"),
}


# Aliases mapping full names, 3-letter codes, and regional tags to canonical 2-letter ISO codes
LANGUAGE_ALIAS_MAP: Dict[str, str] = {
    # Hindi
    "hindi": "hi", "hin": "hi", "hi": "hi", "हिन्दी": "hi", "हिंदी": "hi",
    # Telugu
    "telugu": "te", "tel": "te", "te": "te", "తెలుగు": "te",
    # English
    "english": "en", "eng": "en", "en": "en", "en-in": "en", "en-us": "en", "en-gb": "en",
    # Tamil
    "tamil": "ta", "tam": "ta", "ta": "ta", "தமிழ்": "ta",
    # Kannada
    "kannada": "kn", "kan": "kn", "kn": "kn", "ಕನ್ನಡ": "kn",
    # Malayalam
    "malayalam": "ml", "mal": "ml", "ml": "ml", "മലയാളം": "ml",
    # Marathi
    "marathi": "mr", "mar": "mr", "mr": "mr", "मराठी": "mr",
    # Bengali
    "bengali": "bn", "ben": "bn", "bangla": "bn", "bn": "bn", "বাংলা": "bn",
    # Gujarati
    "gujarati": "gu", "guj": "gu", "gu": "gu", "ગુજરાતી": "gu",
    # Punjabi
    "punjabi": "pa", "pan": "pa", "pa": "pa", "ਪੰਜਾਬੀ": "pa",
    # Urdu
    "urdu": "ur", "urd": "ur", "ur": "ur", "اردو": "ur",
    # Spanish
    "spanish": "es", "esp": "es", "es": "es",
    # French
    "french": "fr", "fra": "fr", "fr": "fr",
    # German
    "german": "de", "deu": "de", "ger": "de", "de": "de",
    # Japanese
    "japanese": "ja", "jpn": "ja", "ja": "ja",
    # Chinese
    "chinese": "zh", "zho": "zh", "chi": "zh", "zh": "zh",
    # Arabic
    "arabic": "ar", "ara": "ar", "ar": "ar",
    # Russian
    "russian": "ru", "rus": "ru", "ru": "ru",
    # Portuguese
    "portuguese": "pt", "por": "pt", "pt": "pt",
    # Italian
    "italian": "it", "ita": "it", "it": "it",
    # Korean
    "korean": "ko", "kor": "ko", "ko": "ko",
}


def normalize_language_code(input_str: Optional[str], default: str = "en") -> str:
    """
    Resolve full language names (e.g. 'hindi', 'eng', 'telugu'),
    3-letter ISO-639-2 codes, or regional tags ('en-US') to canonical 2-letter ISO-639-1 codes.
    """
    if not input_str:
        return default
    clean = input_str.strip().lower()
    if clean in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[clean]
    base = clean.split("-")[0].split("_")[0]
    if base in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[base]
    if base in SUPPORTED_LANGUAGES:
        return base
    return clean[:2] if len(clean) >= 2 else default


@dataclass
class LidDecision:
    """Decision produced by the Language Identification Engine."""
    language_code: str
    language_name: str
    native_name: str
    confidence: float
    detection_source: str  # "spoken_audio" | "unicode_script" | "hybrid" | "tracker_locked"
    is_language_switch: bool = False
    latency_ms: float = 0.0
    top_candidates: List[Tuple[str, str, float]] = field(default_factory=list)  # (code, name, prob)

    def __repr__(self) -> str:
        switch_tag = " [SWITCH DETECTED]" if self.is_language_switch else ""
        return (
            f"LidDecision({self.language_code} ({self.language_name}), "
            f"conf={self.confidence:.1%}, src={self.detection_source}{switch_tag}, "
            f"latency={self.latency_ms:.1f}ms)"
        )
