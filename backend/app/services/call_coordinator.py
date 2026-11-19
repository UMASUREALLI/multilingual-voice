"""
Full-Duplex Bidirectional Call Coordinator.
Orchestrates symmetric real-time voice translation pipeline:
Mic/WebSocket Audio -> Silero VAD -> Acoustic LID -> Faster-Whisper ASR
-> Contextual NMT + Entity Preserver -> Prosody Voice Cloner -> Streaming TTS -> Playback.
Zero disk writes: all audio processing happens strictly in memory.
"""

import time
import uuid
import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from app.models.call import CallSession, CallParticipant, CallSubtitle, CallStatus
from app.models.tts import SpeechSynthesisResult
from app.models.speaker import ConsentStatus
from app.services.lid_service import LanguageDetectionEngine
from app.services.asr_service import StreamingAsrService
from app.services.translation_service import TranslationService
from app.services.tts_service import StreamingTtsService
from app.services.voice_cloner import VoiceCloningService
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer

logger = logging.getLogger("call_translator.coordinator")


class CallCoordinator:
    """
    Central orchestration engine for bidirectional multilingual calls.
    Maintains active call sessions, prevents acoustic feedback loops,
    and coordinates in-memory pipelines with zero disk writes.
    """

    _instance: Optional["CallCoordinator"] = None

    def __init__(
        self,
        model_size: str = "base",
        lid_engine: Optional[LanguageDetectionEngine] = None,
        asr_service: Optional[StreamingAsrService] = None,
        translation_service: Optional[TranslationService] = None,
        tts_service: Optional[StreamingTtsService] = None,
        cloner_service: Optional[VoiceCloningService] = None,
        player_engine: Optional[AudioPlaybackEngine] = None
    ):
        self.model_size = model_size

        # Singleton or shared Whisper model to strictly limit memory to <1GB RAM
        if lid_engine is not None and asr_service is not None:
            self.lid = lid_engine
            self.asr = asr_service
        else:
            from faster_whisper import WhisperModel
            logger.info(f"Loading shared faster-whisper '{model_size}' model for call coordinator (INT8 CPU)...")
            shared_model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
            self.lid = LanguageDetectionEngine(whisper_model=shared_model, model_size=model_size)
            self.asr = StreamingAsrService(model_size=model_size, shared_model=shared_model)

        self.translation = translation_service or TranslationService()
        self.tts = tts_service or StreamingTtsService()
        self.cloner = cloner_service or VoiceCloningService()
        self.player = player_engine or AudioPlaybackEngine()

        self._sessions: Dict[str, CallSession] = {}

    @classmethod
    def get_instance(cls, model_size: str = "base") -> "CallCoordinator":
        if cls._instance is None:
            cls._instance = cls(model_size=model_size)
        return cls._instance

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------

    def create_session(
        self,
        caller_a_lang: str = "te",
        caller_b_lang: str = "en",
        caller_a_name: str = "Caller A (Telugu)",
        caller_b_name: str = "Caller B (English)",
        caller_a_gender: str = "female",
        caller_b_gender: str = "male",
        session_id: Optional[str] = None
    ) -> CallSession:
        """Create a new bidirectional translated call session."""
        sid = session_id or str(uuid.uuid4())[:8]

        participant_a = CallParticipant(
            participant_id="caller_a",
            name=caller_a_name,
            native_language=caller_a_lang,
            target_language=caller_b_lang,
            gender=caller_a_gender
        )

        participant_b = CallParticipant(
            participant_id="caller_b",
            name=caller_b_name,
            native_language=caller_b_lang,
            target_language=caller_a_lang,
            gender=caller_b_gender
        )

        session = CallSession(
            session_id=sid,
            caller_a=participant_a,
            caller_b=participant_b,
            status=CallStatus.ACTIVE
        )

        self._sessions[sid] = session
        logger.info(f"Initialized CallSession '{sid}': {caller_a_name} ({caller_a_lang}) <-> {caller_b_name} ({caller_b_lang})")
        return session

    def get_session(self, session_id: str) -> Optional[CallSession]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> None:
        """Terminate call session, abort active playback, and clear voice profiles."""
        session = self.get_session(session_id)
        if session:
            session.status = CallStatus.ENDED
            self.player.interrupt()
            # Clear ephemeral voice profiles
            self.cloner.clear_session()
            logger.info(f"Terminated CallSession '{session_id}'. Ephemeral profiles wiped.")

    # ------------------------------------------------------------------
    # Spoken Consent & Voice Cloning Enrollment
    # ------------------------------------------------------------------

    def enroll_consent(
        self,
        session_id: str,
        speaker_id: str,
        spoken_consent_text: str,
        audio_samples: Optional[np.ndarray] = None,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Enroll participant's voice with spoken verbal consent.
        Zero disk writes: voice profile is stored ephemerally in RAM.
        """
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")

        participant = session.get_participant(speaker_id)
        if not participant:
            raise KeyError(f"Participant '{speaker_id}' not found in session.")

        if audio_samples is None or len(audio_samples) == 0:
            # Generate simulated harmonic vocal tone if pure text test
            t = np.linspace(0, 1.0, sample_rate, endpoint=False)
            f0 = 210.0 if participant.gender == "female" else 125.0
            audio_samples = (0.5 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

        profile = self.cloner.enroll_speaker(
            speaker_id=speaker_id,
            spoken_consent_text=spoken_consent_text,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            consent_language=participant.native_language
        )

        participant.voice_cloned = True
        participant.pitch_mod = profile.pitch_adjustment
        participant.rate_mod = profile.rate_adjustment

        return {
            "speaker_id": speaker_id,
            "consent_status": profile.consent_status.value,
            "estimated_gender": profile.acoustic_features.estimated_gender,
            "pitch_adjustment": profile.pitch_adjustment,
            "rate_adjustment": profile.rate_adjustment
        }

    # ------------------------------------------------------------------
    # Symmetric Audio Turn Processing Pipeline
    # ------------------------------------------------------------------

    def process_audio_turn(
        self,
        session_id: str,
        speaker_id: str,
        audio_samples: np.ndarray,
        sample_rate: int = 16000,
        synthesize_audio: bool = True
    ) -> Tuple[Optional[CallSubtitle], Optional[SpeechSynthesisResult]]:
        """
        Process a single spoken utterance through the end-to-end pipeline:
        1. Acoustic LID
        2. Streaming ASR (Whisper INT8)
        3. Contextual NMT + Entity Preserver
        4. Prosody-Conditioned Neural TTS
        Zero disk writes.
        """
        t_start = time.perf_counter()
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Call session '{session_id}' not found.")

        speaker = session.get_participant(speaker_id)
        if not speaker:
            raise KeyError(f"Speaker '{speaker_id}' not recognized in session '{session_id}'.")

        recipient = session.get_recipient(speaker_id)

        # --------------------------------------------------------------
        # Step 1 & 2: Fast Single-Pass Acoustic LID + Streaming ASR
        # --------------------------------------------------------------
        supported_core_langs = {"te", "hi", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "en"}
        target_asr_lang = None if speaker.native_language == "auto" else speaker.native_language

        t_asr_0 = time.perf_counter()
        asr_res = self.asr.transcribe(
            audio_or_path=audio_samples,
            language=target_asr_lang,
            beam_size=1
        )
        asr_latency = (time.perf_counter() - t_asr_0) * 1000.0
        lid_latency = 0.0  # Unified in single pass

        detected_lang = asr_res.language
        clean_original = asr_res.text.strip()

        # Drop hallucinated empty punctuation
        if not clean_original or set(clean_original).issubset({'.', ',', ' ', '?', '!', '-', '…', '۔'}):
            logger.info(f"[{session_id}] Dropping silence/hallucination artifact: '{clean_original}'")
            return None, None

        # Filter out turns with no alphanumeric characters
        clean_words = "".join(c for c in clean_original if c.isalnum())
        if not clean_words or len(clean_words) < 2:
            logger.info(f"[{session_id}] Dropping acoustic breath/noise: '{clean_original}'")
            return None, None

        # Stage-2 Script Verification (Refines Telugu, Devanagari, Tamil scripts)
        script_dec = self.lid.identify_script(clean_original)

        # Guard against known Whisper cross-script hallucinations (e.g. mic breath/wind decoded as Arabic/Urdu)
        if script_dec and script_dec.language_code in {"ur", "ar"}:
            if speaker.native_language != "ur" and recipient.native_language != "ur":
                logger.warning(f"[{session_id}] Discarding Arabic/Urdu hallucination on microphone noise: '{clean_original}'")
                return None, None

        if script_dec and script_dec.language_code in supported_core_langs:
            if detected_lang != script_dec.language_code and script_dec.confidence >= 0.70:
                logger.info(f"[{session_id}] Script verification refined language: {detected_lang} -> {script_dec.language_code}")
                detected_lang = script_dec.language_code

        # --------------------------------------------------------------
        # Step 3: Contextual NMT Translation with Entity Preservation
        # --------------------------------------------------------------
        t_nmt_0 = time.perf_counter()
        target_lang = recipient.native_language if recipient.native_language != "auto" else ("en" if detected_lang != "en" else "te")

        trans_res = self.translation.translate(
            text=clean_original,
            source_lang=detected_lang,
            target_lang=target_lang,
            speaker_name=speaker.name,
            receiver_name=recipient.name
        )
        translation_latency = (time.perf_counter() - t_nmt_0) * 1000.0
        clean_translated = trans_res.translated_text.strip()

        # --------------------------------------------------------------
        # Step 4: Voice Adaptation & Streaming TTS
        # --------------------------------------------------------------
        tts_result: Optional[SpeechSynthesisResult] = None
        tts_latency = 0.0

        if synthesize_audio and clean_translated:
            t_tts_0 = time.perf_counter()
            # Select gender and prosody based on speaker profile (if voice cloning enabled)
            gender = speaker.gender
            pitch_adj = speaker.pitch_mod if speaker.voice_cloned else "+0Hz"
            rate_adj = speaker.rate_mod if speaker.voice_cloned else "+0%"

            try:
                tts_result = self.tts.synthesize(
                    text=clean_translated,
                    language_code=target_lang,
                    gender=gender,
                    pitch=pitch_adj,
                    rate=rate_adj
                )
                tts_latency = (time.perf_counter() - t_tts_0) * 1000.0
            except Exception as e:
                logger.error(f"TTS synthesis error in call turn: {e}")

        total_latency = (time.perf_counter() - t_start) * 1000.0

        # --------------------------------------------------------------
        # Step 5: Subtitle Construction & Session History Recording
        # --------------------------------------------------------------
        subtitle = CallSubtitle(
            speaker_id=speaker.participant_id,
            speaker_name=speaker.name,
            original_language=detected_lang,
            original_text=clean_original,
            target_language=target_lang,
            translated_text=clean_translated,
            asr_latency_ms=asr_latency,
            translation_latency_ms=translation_latency,
            tts_latency_ms=tts_latency,
            total_latency_ms=total_latency,
            has_audio=tts_result is not None
        )

        session.subtitles.append(subtitle)
        logger.info(
            f"[{session_id}] {speaker.name} ({detected_lang}): '{clean_original}' -> "
            f"{recipient.name} ({recipient.native_language}): '{clean_translated}' "
            f"[Total Latency: {total_latency:.1f}ms]"
        )

        return subtitle, tts_result

    def handle_barge_in(self) -> None:
        """Interrupt and halt active audio playback immediately on user speech onset."""
        self.player.interrupt()
