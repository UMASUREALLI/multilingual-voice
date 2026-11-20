"""
Multilingual Real-Time Speech-to-Text (ASR) Service.
Consumes in-memory float32 audio buffers with zero disk I/O.
Supports language-constrained decoding, word-level timestamps, anti-hallucination penalties,
and conversational context management.
"""

import logging
import threading
import time
from typing import Optional, Union, List, Dict, Any
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:  # Optional until local Whisper inference is actually needed.
    WhisperModel = None

from app.models.transcription import TranscriptionResult, TranscriptionSegment, WordTiming
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo
from app.services.lid_service import trim_silence, LanguageDetectionEngine

logger = logging.getLogger("call_translator.asr")


class LazyWhisperModel:
    """Thread-safe proxy that loads faster-whisper on the first inference call."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8", cpu_threads: int = 4):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if WhisperModel is None:
                        raise RuntimeError(
                            "faster-whisper is not installed. Install the full requirements.txt "
                            "to enable offline speech recognition."
                        )
                    logger.info(
                        "Initializing faster-whisper '%s' (%s on %s)...",
                        self.model_size,
                        self.compute_type,
                        self.device,
                    )
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=self.cpu_threads,
                    )
        return self._model

    def transcribe(self, *args, **kwargs):
        return self._get_model().transcribe(*args, **kwargs)

    def detect_language(self, *args, **kwargs):
        return self._get_model().detect_language(*args, **kwargs)

# BCP-47 locale mapping for high-accuracy native-script Neural STT across Indian & Global languages
LANG_TO_GOOGLE_LOCALE: Dict[str, str] = {
    # Indian Languages
    "te": "te-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "ur": "ur-IN",
    "or": "or-IN",
    "as": "as-IN",
    "ne": "ne-NP",
    "si": "si-LK",
    # Global Languages
    "en": "en-IN",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ru": "ru-RU",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "tr": "tr-TR",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "vi": "vi-VN",
    "th": "th-TH",
    "id": "id-ID",
    "ms": "ms-MY",
    "sv": "sv-SE",
    "uk": "uk-UA",
    "fa": "fa-IR",
    "he": "he-IL",
}


class ConversationalContextManager:
    """
    Tracks recent conversational history to provide Whisper with acoustic/semantic context.
    Helps correctly transcribe proper nouns, names, and technical terms across turns.
    """

    def __init__(self, max_history_turns: int = 4):
        self.max_history_turns = max_history_turns
        self._history: List[str] = []

    def add_turn(self, text: str) -> None:
        clean = text.strip()
        if clean and len(clean) > 2 and not set(clean).issubset({'.', ',', ' ', '?', '!', '-', '…'}):
            self._history.append(clean)
            if len(self._history) > self.max_history_turns:
                self._history.pop(0)

    def get_context_prompt(self) -> Optional[str]:
        if not self._history:
            return None
        # Format rolling context as initial prompt
        return "Previous conversation: " + " ".join(self._history[-3:])

    def clear(self) -> None:
        self._history.clear()


class StreamingAsrService:
    """
    High-Performance In-Memory Multilingual Speech Recognition Engine.
    Powered by faster-whisper CTranslate2 (INT8 CPU).
    """

    _instance: Optional["StreamingAsrService"] = None

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        shared_model: Optional[Any] = None
    ):
        self.model_size = model_size
        self.context_manager = ConversationalContextManager()

        self.model = shared_model or LazyWhisperModel(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    @classmethod
    def get_instance(cls, model_size: str = "base") -> "StreamingAsrService":
        """Singleton accessor to ensure a single model is shared across the application."""
        if cls._instance is None:
            cls._instance = StreamingAsrService(model_size=model_size)
        return cls._instance

    @staticmethod
    def _transcribe_google_neural(audio: np.ndarray, sample_rate: int, language_code: str) -> tuple:
        """
        In-memory Neural Speech-to-Text via SpeechRecognition (Google Speech API).
        Returns: (text_or_none, status) where status is 'ok', 'no_speech', or 'error'.
        Zero disk writes.
        """
        try:
            import speech_recognition as sr_lib
        except ImportError:
            return None, "error"

        clipped = np.clip(audio, -1.0, 1.0)
        pcm16_bytes = (clipped * 32767.0).astype(np.int16).tobytes()
        audio_data = sr_lib.AudioData(pcm16_bytes, sample_rate, 2)
        locale = LANG_TO_GOOGLE_LOCALE.get(language_code, f"{language_code}-{language_code.upper()}")

        recognizer = sr_lib.Recognizer()
        try:
            text = recognizer.recognize_google(audio_data, language=locale)
            if text and text.strip():
                return text.strip(), "ok"
            return "", "no_speech"
        except sr_lib.UnknownValueError:
            return "", "no_speech"
        except Exception as e:
            logger.warning(f"Neural STT unavailable ({e}); falling back to offline Faster-Whisper.")
            return None, "error"

    def transcribe(
        self,
        audio_or_path: Union[str, np.ndarray],
        language: Optional[str] = None,
        language_tracker: Optional[Any] = None,
        initial_prompt: Optional[str] = None,
        beam_size: int = 1,
        word_timestamps: bool = False,
        trim_audio: bool = True,
        prefer_neural_stt: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio (in-memory numpy array or file path) into native-script text.
        Args:
            audio_or_path: np.ndarray (float32 at 16kHz) or file path.
            language: Target language code (e.g., 'te', 'hi', 'ta', 'en'). If None, auto-detected via VoxLingua107 ECAPA.
            language_tracker: Optional DynamicLanguageTracker instance to lock session language and prevent turn-by-turn jitter.
            initial_prompt: Optional conversational context prompt (used by Whisper fallback).
            beam_size: 1 for fast greedy decode, 2-3 for beam search.
            word_timestamps: Whether to extract per-word start/end timing tokens (routes to Whisper).
            trim_audio: Whether to strip leading/trailing silence before decoding.
            prefer_neural_stt: Use high-accuracy native-script Neural STT (with offline Whisper fallback).
        """
        t0 = time.perf_counter()

        # Handle input formats
        if isinstance(audio_or_path, str):
            import soundfile as sf
            audio, sr = sf.read(audio_or_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)
            if sr != 16000 and len(audio) > 0:
                import scipy.signal
                audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
            duration_sec = len(audio) / 16000.0
        else:
            audio = np.asarray(audio_or_path, dtype=np.float32).reshape(-1)
            duration_sec = len(audio) / 16000.0

        if len(audio) == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return TranscriptionResult(
                text="",
                language=language or "en",
                language_name="English",
                language_probability=0.0,
                duration_sec=0.0,
                latency_ms=elapsed,
                rtf=0.0
            )

        # 1. Silence trimming
        if trim_audio and isinstance(audio, np.ndarray):
            speech_audio, start_sec, dur_sec = trim_silence(audio, sample_rate=16000)
            if len(speech_audio) >= int(16000 * 0.25):
                audio = speech_audio
                duration_sec = dur_sec

        # 2. Automatic Language Detection via Session Tracker or VoxLingua107 ECAPA-TDNN (107 languages)
        resolved_lang = language
        lang_prob = 1.0
        if resolved_lang is None:
            if language_tracker is not None:
                lid_dec = language_tracker.process_turn(audio=audio, sample_rate=16000)
                if getattr(language_tracker, "is_locked", False):
                    resolved_lang = lid_dec.language_code
                    lang_prob = float(lid_dec.confidence)

            if resolved_lang is None:
                lid_engine = LanguageDetectionEngine(whisper_model=self.model, model_size=self.model_size)
                lid_dec = lid_engine.identify_spoken(audio, sample_rate=16000, trim_audio=False)
                resolved_lang = lid_dec.language_code
                lang_prob = float(lid_dec.confidence)

        # 3. Primary Path: High-Accuracy Native-Script Neural STT (Telugu, Hindi, Tamil, Kannada, Malayalam, English, Global)
        if prefer_neural_stt and not word_timestamps:
            # Peak-normalize copy for clear recognition on quiet microphones
            stt_audio = audio
            peak = float(np.max(np.abs(stt_audio))) if len(stt_audio) > 0 else 0.0
            if peak > 0.003:
                stt_audio = (stt_audio / peak * 0.90).astype(np.float32)

            neural_text, status = self._transcribe_google_neural(stt_audio, 16000, resolved_lang)
            if status == "ok" and neural_text:
                elapsed_sec = time.perf_counter() - t0
                latency_ms = elapsed_sec * 1000.0
                rtf = elapsed_sec / max(0.1, duration_sec)
                self.context_manager.add_turn(neural_text)

                if language_tracker is not None:
                    final_dec = language_tracker.process_turn(transcript=neural_text)
                    resolved_lang = final_dec.language_code
                    lang_prob = float(final_dec.confidence)

                lang_info = SUPPORTED_LANGUAGES.get(
                    resolved_lang,
                    LanguageInfo(resolved_lang, resolved_lang.title(), resolved_lang)
                )
                seg = TranscriptionSegment(
                    id=0,
                    start=0.0,
                    end=round(duration_sec, 2),
                    text=neural_text,
                    words=[]
                )
                return TranscriptionResult(
                    text=neural_text,
                    language=resolved_lang,
                    language_name=lang_info.name,
                    language_probability=lang_prob,
                    duration_sec=duration_sec,
                    latency_ms=latency_ms,
                    rtf=rtf,
                    segments=[seg],
                    words=[]
                )
            elif status == "no_speech":
                # Pure silence / unintelligible breath — return empty cleanly without Whisper hallucination
                elapsed_sec = time.perf_counter() - t0
                lang_info = SUPPORTED_LANGUAGES.get(
                    resolved_lang,
                    LanguageInfo(resolved_lang, resolved_lang.title(), resolved_lang)
                )
                return TranscriptionResult(
                    text="",
                    language=resolved_lang,
                    language_name=lang_info.name,
                    language_probability=lang_prob,
                    duration_sec=duration_sec,
                    latency_ms=elapsed_sec * 1000.0,
                    rtf=elapsed_sec / max(0.1, duration_sec),
                    segments=[],
                    words=[]
                )

        # 4. Offline Fallback / Word-Timestamps Path: Local Faster-Whisper (INT8 CPU)
        # Note: repetition_penalty=1.0 and no_repeat_ngram_size=0 prevent UTF-8 3-byte Indic script corruption.
        prompt = initial_prompt
        try:
            segments, info = self.model.transcribe(
                audio,
                language=resolved_lang,
                initial_prompt=prompt,
                beam_size=beam_size,
                best_of=1,
                word_timestamps=word_timestamps,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                condition_on_previous_text=False,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6
            )
        except Exception as exc:
            logger.warning("Offline ASR unavailable: %s", exc)
            elapsed_sec = time.perf_counter() - t0
            lang_info = SUPPORTED_LANGUAGES.get(
                resolved_lang,
                LanguageInfo(resolved_lang, resolved_lang.title(), resolved_lang),
            )
            return TranscriptionResult(
                text="",
                language=resolved_lang,
                language_name=lang_info.name,
                language_probability=lang_prob,
                duration_sec=duration_sec,
                latency_ms=elapsed_sec * 1000.0,
                rtf=elapsed_sec / max(0.1, duration_sec),
                segments=[],
                words=[],
            )

        segs_list = list(segments)
        elapsed_sec = time.perf_counter() - t0
        latency_ms = elapsed_sec * 1000.0
        rtf = elapsed_sec / max(0.1, duration_sec)

        # Parse segments and word tokens
        parsed_segments: List[TranscriptionSegment] = []
        all_words: List[WordTiming] = []
        full_text_pieces: List[str] = []

        for s in segs_list:
            s_text = s.text.strip()
            if s_text:
                full_text_pieces.append(s_text)

            s_words: List[WordTiming] = []
            if word_timestamps and s.words:
                for w in s.words:
                    wt = WordTiming(
                        word=w.word.strip(),
                        start=round(w.start, 2),
                        end=round(w.end, 2),
                        probability=round(w.probability, 4)
                    )
                    s_words.append(wt)
                    all_words.append(wt)

            parsed_segments.append(
                TranscriptionSegment(
                    id=s.id,
                    start=round(s.start, 2),
                    end=round(s.end, 2),
                    text=s_text,
                    words=s_words
                )
            )

        full_text = " ".join(full_text_pieces).strip()

        # Update conversational context
        if full_text:
            self.context_manager.add_turn(full_text)
            if language_tracker is not None:
                final_dec = language_tracker.process_turn(transcript=full_text)
                resolved_lang = final_dec.language_code
                lang_prob = float(final_dec.confidence)

        final_lang = resolved_lang or info.language or "en"
        final_prob = lang_prob if language is None else float(info.language_probability)
        lang_info = SUPPORTED_LANGUAGES.get(
            final_lang,
            LanguageInfo(final_lang, final_lang.title(), final_lang)
        )

        return TranscriptionResult(
            text=full_text,
            language=final_lang,
            language_name=lang_info.name,
            language_probability=final_prob,
            duration_sec=duration_sec,
            latency_ms=latency_ms,
            rtf=rtf,
            segments=parsed_segments,
            words=all_words
        )


# =========================================================================
# Backwards-Compatible Legacy Function (Preserved for existing test scripts)
# =========================================================================
_default_asr_service: Optional[StreamingAsrService] = None


def get_asr_service() -> StreamingAsrService:
    global _default_asr_service
    if _default_asr_service is None:
        _default_asr_service = StreamingAsrService.get_instance(model_size="base")
    return _default_asr_service


def transcribe_audio(audio_path: str) -> dict:
    """
    Legacy wrapper preserved for test_mic_asr.py and other scripts.
    Accepts an audio file path and returns dict with text, language, and probability.
    """
    service = get_asr_service()
    res = service.transcribe(audio_path)
    return {
        "text": res.text,
        "language": res.language,
        "language_probability": round(res.language_probability, 4)
    }
