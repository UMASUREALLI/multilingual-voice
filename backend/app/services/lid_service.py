"""
Automatic Real-Time Language Detection (LID) Engine.
Combines silence-trimmed acoustic analysis, anti-hallucination token decoding,
and hierarchical consensus to accurately identify Telugu, Hindi, Tamil, English, and more.
"""

import os
import time
import logging
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, LidDecision, normalize_language_code

logger = logging.getLogger("call_translator.lid")


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -45.0,
    pad_ms: int = 350
) -> Tuple[np.ndarray, float, float]:
    """
    Strips leading and trailing silence from an audio array based on energy.
    Prevents feeding room noise to Whisper which causes false English predictions.
    Returns: (trimmed_audio, start_offset_sec, trimmed_duration_sec)
    """
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=np.float32), 0.0, 0.0

    flat_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame_size = 512  # 32ms
    n_frames = len(flat_audio) // frame_size
    if n_frames == 0:
        return flat_audio, 0.0, len(flat_audio) / sample_rate

    active_indices = []
    for i in range(n_frames):
        chunk = flat_audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        db = 20.0 * np.log10(max(rms, 1e-6))
        if db >= threshold_db:
            active_indices.append(i)

    if not active_indices:
        return flat_audio, 0.0, len(flat_audio) / sample_rate

    pad_frames = int((pad_ms / 1000.0) * (sample_rate / frame_size))
    first_frame = max(0, active_indices[0] - pad_frames)
    last_frame = min(n_frames, active_indices[-1] + pad_frames + 1)

    start_sec = (first_frame * frame_size) / sample_rate
    trimmed = flat_audio[first_frame * frame_size : last_frame * frame_size]
    return trimmed, start_sec, len(trimmed) / sample_rate


class LanguageDetectionEngine:
    """
    High-Accuracy Multilingual Language Detection Engine (107+ Languages).
    Primary Acoustic LID: SpeechBrain VoxLingua107 ECAPA-TDNN (~85MB, CPU-fast,
    near-100% accuracy across Telugu, Hindi, Tamil, Kannada, Malayalam, English,
    Spanish, French, and 100+ global languages).
    Fallback Acoustic LID: Faster-Whisper acoustic language logits.
    Stage-2 Verification: Deterministic Unicode Script Analyzer.
    """

    _shared_ecapa_model = None
    _ecapa_load_failed = False

    def __init__(self, whisper_model=None, model_size: str = "base", default_language: str = "en", use_ecapa: bool = True):
        self.default_language = default_language
        self.model_size = model_size
        self._model = whisper_model
        self.use_ecapa = use_ecapa

    @classmethod
    def get_ecapa_model(cls):
        """Singleton loader for SpeechBrain VoxLingua107 ECAPA-TDNN (cached on disk, runs in RAM)."""
        if cls._shared_ecapa_model is not None:
            return cls._shared_ecapa_model
        if cls._ecapa_load_failed:
            return None

        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            savedir = os.path.join(backend_dir, "pretrained_models", "lang-id-voxlingua107-ecapa")
            logger.info("Loading SpeechBrain VoxLingua107 ECAPA-TDNN acoustic LID model (107 languages)...")
            cls._shared_ecapa_model = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir=savedir,
                local_strategy=LocalStrategy.COPY
            )
            try:
                cls._shared_ecapa_model.hparams.label_encoder.ignore_len()
            except Exception:
                pass
            logger.info("VoxLingua107 ECAPA-TDNN LID model ready.")
            return cls._shared_ecapa_model
        except Exception as e:
            logger.warning(f"VoxLingua107 ECAPA LID unavailable ({e}); falling back to Whisper acoustic LID.")
            cls._ecapa_load_failed = True
            return None

    def _get_model(self):
        """Lazy load or reuse shared Whisper model instance (fallback)."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed; acoustic LID fallback is unavailable"
                ) from exc
            logger.info(f"Loading faster-whisper '{self.model_size}' model (INT8 CPU)...")
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def _identify_with_ecapa(self, speech_audio: np.ndarray, sample_rate: int, t0: float) -> Optional[LidDecision]:
        """Run VoxLingua107 ECAPA-TDNN on in-memory float32 audio (16kHz)."""
        ecapa = self.get_ecapa_model()
        if ecapa is None:
            return None

        import torch
        audio_16k = speech_audio
        if sample_rate != 16000 and len(audio_16k) > 0:
            import scipy.signal
            num_samples = int(len(audio_16k) * 16000 / sample_rate)
            audio_16k = scipy.signal.resample(audio_16k, num_samples).astype(np.float32)

        # Evaluate up to 15 seconds of active speech for fast, stable accuracy
        audio_16k = audio_16k[: min(len(audio_16k), 16000 * 15)]
        wav_tensor = torch.from_numpy(np.ascontiguousarray(audio_16k, dtype=np.float32)).unsqueeze(0)

        with torch.no_grad():
            out_prob, score, index, text_lab = ecapa.classify_batch(wav_tensor)

        elapsed = (time.perf_counter() - t0) * 1000.0
        probs = torch.exp(out_prob[0]).cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:8]

        # Normalize closely related acoustic dialect codes (e.g. Scots 'sco' -> English 'en')
        alias_map = {"sco": "en", "iw": "he", "jw": "jv"}

        candidates: List[Tuple[str, str, float]] = []
        seen_codes = set()
        for idx in top_indices:
            lbl = ecapa.hparams.label_encoder.decode_ndim(int(idx))
            raw_code = str(lbl).split(":")[0].strip().lower()
            c_code = alias_map.get(raw_code, raw_code)
            c_prob = float(probs[idx])
            if c_code in seen_codes:
                continue
            seen_codes.add(c_code)
            info = SUPPORTED_LANGUAGES.get(c_code, LanguageInfo(c_code, c_code.title(), c_code))
            candidates.append((c_code, info.name, c_prob))

        raw_top = str(text_lab[0]).split(":")[0].strip().lower()
        best_code = alias_map.get(raw_top, raw_top)
        best_conf = float(torch.exp(score[0]).item())

        # If raw_top is not in our supported languages, pick the top supported candidate
        supported_candidates = [c for c in candidates if c[0] in SUPPORTED_LANGUAGES]
        if best_code not in SUPPORTED_LANGUAGES and supported_candidates:
            best_code = supported_candidates[0][0]
            best_conf = supported_candidates[0][2]

        return self._build_decision(
            lang_code=best_code,
            confidence=best_conf,
            source="voxlingua107_ecapa",
            latency_ms=elapsed,
            candidates=candidates
        )

    def identify_spoken(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        trim_audio: bool = True
    ) -> LidDecision:
        """
        Identify the language spoken in an audio chunk across 107+ languages.
        Strips silence first, then runs VoxLingua107 ECAPA-TDNN (with Whisper fallback).
        """
        t0 = time.perf_counter()
        if audio is None or len(audio) == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return self._build_decision(self.default_language, 0.0, "empty_audio", elapsed)

        flat_audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        # 1. Strip leading and trailing silence
        if trim_audio:
            speech_audio, start_sec, speech_dur = trim_silence(flat_audio, sample_rate=sample_rate)
        else:
            speech_audio = flat_audio

        # Minimum speech duration check (at least 300ms)
        if len(speech_audio) < int(sample_rate * 0.30):
            elapsed = (time.perf_counter() - t0) * 1000.0
            return self._build_decision(self.default_language, 0.25, "insufficient_speech", elapsed)

        # 2. Primary: VoxLingua107 ECAPA-TDNN (107 languages, ~100% Telugu/Hindi/Tamil/Kannada/Malayalam/Global)
        if self.use_ecapa:
            try:
                ecapa_dec = self._identify_with_ecapa(speech_audio, sample_rate, t0)
                if ecapa_dec is not None:
                    return ecapa_dec
            except Exception as e:
                logger.warning(f"ECAPA LID inference failed ({e}); falling back to Whisper LID.")

        # 3. Fallback: Faster-Whisper Acoustic Logits
        try:
            model = self._get_model()
            # Detect language on actual speech (up to 30s)
            lang_code, confidence, all_probs = model.detect_language(
                speech_audio[: min(len(speech_audio), sample_rate * 30)]
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            supported_probs = [(c_code, float(c_prob)) for c_code, c_prob in all_probs if c_code in SUPPORTED_LANGUAGES]
            candidates: List[Tuple[str, str, float]] = []
            for c_code, c_prob in (supported_probs[:8] if supported_probs else all_probs[:8]):
                info = SUPPORTED_LANGUAGES.get(c_code, LanguageInfo(c_code, c_code.title(), c_code))
                candidates.append((c_code, info.name, float(c_prob)))

            if supported_probs:
                supported_probs.sort(key=lambda x: x[1], reverse=True)
                best_code = supported_probs[0][0]
                best_conf = supported_probs[0][1]
            else:
                best_code = str(lang_code)
                best_conf = float(confidence)

            return self._build_decision(
                lang_code=best_code,
                confidence=best_conf,
                source="spoken_acoustic",
                latency_ms=elapsed,
                candidates=candidates
            )
        except Exception as e:
            logger.error(f"Error in spoken LID: {e}")
            elapsed = (time.perf_counter() - t0) * 1000.0
            return self._build_decision(self.default_language, 0.0, "spoken_error", elapsed)

    def identify_script(self, text: str) -> Optional[LidDecision]:
        """
        Stage-2 Deterministic Unicode Script Identifier.
        Analyzes Unicode blocks: Telugu (0x0C00-0x0C7F), Devanagari (0x0900-0x097F), Tamil (0x0B80-0x0BFF), etc.
        """
        t0 = time.perf_counter()
        clean = text.strip()
        if not clean:
            return None

        script_counts: Dict[str, int] = {}
        latin_count = 0

        for ch in clean:
            cp = ord(ch)
            if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
                latin_count += 1
                continue

            for code, lang in SUPPORTED_LANGUAGES.items():
                if lang.unicode_range and code != "en":
                    lo, hi = lang.unicode_range
                    if lo <= cp <= hi:
                        script_counts[code] = script_counts.get(code, 0) + 1

        elapsed = (time.perf_counter() - t0) * 1000.0

        if script_counts:
            best_code = max(script_counts, key=script_counts.get)

            # Disambiguate Devanagari (Hindi vs Marathi vs Nepali)
            if best_code in {"hi", "mr", "ne"}:
                if "\u0933" in clean or any(w in clean for w in ("आहे", "नाही", "तुम्ही", "माझं")):
                    best_code = "mr"
                elif any(w in clean for w in ("छ", "छैन", "गर्न", "तपाईं")):
                    best_code = "ne"
                else:
                    best_code = "hi"

            # Disambiguate Bengali vs Assamese
            elif best_code in {"bn", "as"}:
                if "\u09F0" in clean or "\u09F1" in clean:
                    best_code = "as"
                else:
                    best_code = "bn"

            confidence = 0.96 if latin_count == 0 else 0.88
            return self._build_decision(best_code, confidence, "unicode_script", elapsed)

        elif latin_count >= 3:
            return self._build_decision("en", 0.92, "unicode_script", elapsed)

        return None

    def identify(self, text: str) -> LidDecision:
        """Convenience method to identify language from text string with English fallback."""
        dec = self.identify_script(text)
        if dec is not None:
            return dec
        return self._build_decision("en", 0.85, "text_default", 0.0)

    def full_speech_and_language_analysis(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[LidDecision, str]:
        """
        Hierarchical Speech & Language Analysis:
        1. Run acoustic spoken LID on trimmed speech.
        2. Constrain the ASR decoder to the acoustic prediction (preventing hallucinatory script loops).
        3. Cross-verify with Unicode script identification.
        Returns: (final_lid_decision, transcribed_text)
        """
        # Step 1: Silence trimming
        speech_audio, _, _ = trim_silence(audio, sample_rate=sample_rate)
        if len(speech_audio) < int(sample_rate * 0.30):
            return self.identify_spoken(audio, sample_rate), ""

        # Step 2: Spoken acoustic detection first
        spoken_decision = self.identify_spoken(speech_audio, sample_rate, trim_audio=False)

        model = self._get_model()

        # Step 3: Transcribe with language constrained to acoustic decision if confidence >= 0.60
        # If acoustic confidence is strong (e.g. Telugu 79.5%), we tell Whisper to decode in Telugu!
        # This prevents Whisper from hallucinating Devanagari or Tamil repetition loops!
        target_decode_lang = spoken_decision.language_code if spoken_decision.confidence >= 0.60 else None

        segments, info = model.transcribe(
            speech_audio,
            language=target_decode_lang,
            beam_size=1,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            condition_on_previous_text=False,
            temperature=0.0
        )
        segs = list(segments)
        transcript = " ".join(s.text.strip() for s in segs).strip()

        # Step 4: Script analysis
        script_decision = self.identify_script(transcript)

        # Step 5: Decision Resolution
        # The Acoustic LID is authoritative for spoken language.
        final_decision = spoken_decision

        # If transcript contains valid script matching or refining acoustic decision, reflect it
        if script_decision is not None and script_decision.language_code == spoken_decision.language_code:
            final_decision = LidDecision(
                language_code=spoken_decision.language_code,
                language_name=spoken_decision.language_name,
                native_name=spoken_decision.native_name,
                confidence=min(0.99, max(spoken_decision.confidence, script_decision.confidence)),
                detection_source="acoustic_and_script_verified",
                latency_ms=spoken_decision.latency_ms,
                top_candidates=spoken_decision.top_candidates
            )

        return final_decision, transcript

    def _build_decision(
        self,
        lang_code: str,
        confidence: float,
        source: str,
        latency_ms: float,
        candidates: Optional[List[Tuple[str, str, float]]] = None
    ) -> LidDecision:
        info = SUPPORTED_LANGUAGES.get(
            lang_code,
            LanguageInfo(lang_code, lang_code.title(), lang_code)
        )
        return LidDecision(
            language_code=info.code,
            language_name=info.name,
            native_name=info.native_name,
            confidence=confidence,
            detection_source=source,
            is_language_switch=False,
            latency_ms=round(latency_ms, 2),
            top_candidates=candidates or [(info.code, info.name, confidence)]
        )


class DynamicLanguageTracker:
    """
    Session-level Language State Tracker with Hysteresis & Anti-Jitter Guards.
    Locks caller's active language and handles switching mid-call without fluctuating on quiet turns.
    """

    def __init__(
        self,
        engine: LanguageDetectionEngine,
        initial_language: str = "en",
        switch_confidence_threshold: float = 0.72,
        consecutive_turns_for_switch: int = 2
    ):
        self.engine = engine
        self.switch_threshold = float(switch_confidence_threshold)
        self.consecutive_required = int(consecutive_turns_for_switch)

        norm_init = (initial_language or "auto").strip().lower()
        if norm_init in {"auto", "none", ""}:
            self._is_locked = False
            self._current_decision: LidDecision = engine._build_decision(
                "auto", 0.0, "initial_auto", 0.0
            )
        else:
            self._is_locked = True
            self._current_decision: LidDecision = engine._build_decision(
                norm_init, 1.0, "initial_default", 0.0
            )

        self._pending_switch_code: Optional[str] = None
        self._pending_count: int = 0
        self._history: List[LidDecision] = []
        self._listeners: List[Callable[[str, str, float], None]] = []

    @property
    def is_locked(self) -> bool:
        """Returns True once a specific language has been established for the speaker session."""
        return self._is_locked

    @property
    def current_language(self) -> str:
        return self._current_decision.language_code

    @property
    def current_decision(self) -> LidDecision:
        return self._current_decision

    def register_switch_listener(self, cb: Callable[[str, str, float], None]) -> None:
        self._listeners.append(cb)

    def lock_language(self, language_code: str, confidence: float = 1.0, source: str = "verified_speech") -> LidDecision:
        """Explicitly lock the speaker's session language upon verifying real transcribed speech."""
        norm_code = normalize_language_code(language_code, default="en")
        dec = self.engine._build_decision(norm_code, confidence, source, 0.0)
        self._is_locked = True
        self._current_decision = dec
        self._pending_switch_code = None
        self._pending_count = 0
        self._history.append(dec)
        logger.info(
            f"🔒 Speaker session language locked: {dec.language_name} [{dec.language_code}] (conf={confidence:.1%}, source={source})"
        )
        return dec

    def process_turn(
        self,
        audio: Optional[np.ndarray] = None,
        transcript: Optional[str] = None,
        sample_rate: int = 16000
    ) -> LidDecision:
        candidates: List[LidDecision] = []

        spoken_decision: Optional[LidDecision] = None
        if audio is not None and len(audio) > 0:
            spoken_decision = self.engine.identify_spoken(audio, sample_rate=sample_rate)
            candidates.append(spoken_decision)

        script_decision: Optional[LidDecision] = None
        if transcript and transcript.strip():
            script_decision = self.engine.identify_script(transcript)
            if script_decision is not None:
                candidates.append(script_decision)

        if not candidates:
            return self._current_decision

        # Case A: Initial mode was "auto" and speaker language is not yet locked
        if not self._is_locked:
            # 1. Script verified in transcript -> lock immediately
            if script_decision is not None and script_decision.confidence >= 0.70:
                return self.lock_language(script_decision.language_code, script_decision.confidence, "script_verified")

            # 2. High-confidence spoken audio (>= 0.88 on >= 0.8s) -> lock
            if (
                spoken_decision is not None
                and spoken_decision.language_code not in {"auto", "unknown"}
                and spoken_decision.confidence >= 0.88
                and audio is not None
                and len(audio) >= int(sample_rate * 0.80)
            ):
                return self.lock_language(spoken_decision.language_code, spoken_decision.confidence, "acoustic_verified")

            # 3. Otherwise return tentative spoken decision without locking yet (prevents locking on breath/noise)
            if spoken_decision is not None and spoken_decision.language_code not in {"auto", "unknown"}:
                return spoken_decision
            return self._current_decision

        # Case B: Session is already locked to a language (apply continuity prior & hysteresis)
        curr_lang = self._current_decision.language_code

        # Continuity Prior: If spoken decision top candidates contain curr_lang, apply inertia bonus (+0.25)
        # to prevent acoustic jitter between phonetically close languages (e.g. Tamil vs Kannada vs Telugu)
        if spoken_decision and spoken_decision.top_candidates:
            cand_dict = {c[0]: c[2] for c in spoken_decision.top_candidates}
            curr_prob = cand_dict.get(curr_lang, 0.0)
            top_prob = spoken_decision.confidence
            top_code = spoken_decision.language_code

            if top_code != curr_lang and (curr_prob + 0.25) >= top_prob:
                spoken_decision = LidDecision(
                    language_code=self._current_decision.language_code,
                    language_name=self._current_decision.language_name,
                    native_name=self._current_decision.native_name,
                    confidence=min(0.99, max(self._current_decision.confidence, curr_prob + 0.25)),
                    detection_source="tracker_continuity_held",
                    is_language_switch=False,
                    latency_ms=spoken_decision.latency_ms,
                    top_candidates=spoken_decision.top_candidates
                )

        # Primary decision: prefer script if confident, else spoken
        if script_decision is not None and script_decision.confidence >= 0.85:
            decision = script_decision
        else:
            decision = spoken_decision or candidates[0]

        candidate_lang = decision.language_code

        if candidate_lang != curr_lang:
            # Immediate switch requires high confidence (>= max(0.80, switch_threshold))
            immediate_threshold = max(0.80, self.switch_threshold)
            if decision.confidence >= immediate_threshold:
                self._apply_switch(curr_lang, decision)
                return decision
            else:
                if self._pending_switch_code == candidate_lang:
                    self._pending_count += 1
                else:
                    self._pending_switch_code = candidate_lang
                    self._pending_count = 1

                if self._pending_count >= self.consecutive_required:
                    self._apply_switch(curr_lang, decision)
                    return decision
                else:
                    held_dec = LidDecision(
                        language_code=self._current_decision.language_code,
                        language_name=self._current_decision.language_name,
                        native_name=self._current_decision.native_name,
                        confidence=self._current_decision.confidence,
                        detection_source="tracker_locked",
                        is_language_switch=False,
                        latency_ms=decision.latency_ms,
                        top_candidates=decision.top_candidates
                    )
                    self._history.append(held_dec)
                    return held_dec
        else:
            self._pending_switch_code = None
            self._pending_count = 0
            self._current_decision = decision
            self._history.append(decision)
            if len(self._history) > 50:
                self._history.pop(0)
            return decision

    def _apply_switch(self, old_lang: str, new_decision: LidDecision) -> None:
        new_decision.is_language_switch = True
        self._current_decision = new_decision
        self._pending_switch_code = None
        self._pending_count = 0

        logger.info(
            f"🔄 Language Switch: {old_lang} -> {new_decision.language_code} "
            f"({new_decision.language_name}, conf={new_decision.confidence:.1%})"
        )

        for cb in self._listeners:
            try:
                cb(old_lang, new_decision.language_code, new_decision.confidence)
            except Exception as e:
                logger.error(f"Error in language switch listener: {e}")
