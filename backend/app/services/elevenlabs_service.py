"""
ElevenLabs Instant Voice Cloning & Neural TTS Service.

Integrates with ElevenLabs API to:
1. Instant Voice Clone: Upload 10-30s sample of the user's voice to create an authentic clone.
2. Multilingual Synthesis: Generate English, Telugu, Hindi, etc., matching the exact vocal timbre.
"""

import os
import io
import logging
from typing import Optional, Iterator
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class ElevenLabsVoiceService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self._client = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            logger.info("ElevenLabs client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            self._client = None

    def is_configured(self) -> bool:
        return self._client is not None

    def clone_voice_from_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        voice_name: str = "User_Clone"
    ) -> Optional[str]:
        """
        Creates an Instant Voice Clone from in-memory audio array.
        Returns the cloned voice_id.
        """
        if not self._client:
            raise RuntimeError("ElevenLabs client is not configured. Provide an API key.")

        # Convert numpy audio to in-memory WAV buffer
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        wav_buf.seek(0)
        wav_bytes = wav_buf.read()

        logger.info(f"Uploading {len(wav_bytes)} bytes voice sample to ElevenLabs Instant Voice Clone...")
        try:
            voice_response = self._client.voices.ivc.create(
                name=voice_name,
                files=[("sample.wav", wav_bytes, "audio/wav")],
                remove_background_noise=True,
                description="Instant voice clone from live phone/mic session."
            )
            voice_id = voice_response.voice_id
            logger.info(f"Instant Voice Clone created successfully! Voice ID: {voice_id}")
            return voice_id
        except Exception as e:
            logger.error(f"Error creating ElevenLabs Instant Voice Clone: {e}")
            raise

    def synthesize_cloned(
        self,
        text: str,
        voice_id: str,
        language_code: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128"
    ) -> bytes:
        """
        Synthesizes translated text using the cloned voice.
        Returns raw audio bytes (MP3 or WAV).
        """
        if not self._client:
            raise RuntimeError("ElevenLabs client is not configured. Provide an API key.")

        logger.info(f"Synthesizing text with ElevenLabs voice {voice_id}: '{text}'")
        try:
            audio_generator: Iterator[bytes] = self._client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
                output_format=output_format,
                language_code=language_code
            )
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            raise
