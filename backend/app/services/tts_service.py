"""
Streaming Text-to-Speech (TTS) Service using Edge-TTS with Zero Disk Writes.
Converts translated text into natural neural speech in memory across Indian and global languages.
Supports real-time audio chunk streaming, pitch/rate modulation for voice cloning,
and in-memory LRU caching for ultra-low latency.
"""

import io
import time
import asyncio
import logging
from typing import Optional, Dict, List, Tuple, AsyncIterator
from collections import OrderedDict
import concurrent.futures

import numpy as np
import soundfile as sf
import edge_tts

from app.models.tts import VoiceProfile, SpeechSynthesisRequest, SpeechSynthesisResult

logger = logging.getLogger(__name__)


# Standard Voice Registry for Multilingual Call Translation
DEFAULT_VOICE_REGISTRY: Dict[str, Dict[str, VoiceProfile]] = {
    # Indian Languages
    "te": {
        "female": VoiceProfile("te-IN-ShrutiNeural", "Shruti", "female", "te", "te-IN", "Telugu Female Neural"),
        "male": VoiceProfile("te-IN-MohanNeural", "Mohan", "male", "te", "te-IN", "Telugu Male Neural"),
    },
    "hi": {
        "female": VoiceProfile("hi-IN-SwaraNeural", "Swara", "female", "hi", "hi-IN", "Hindi Female Neural"),
        "male": VoiceProfile("hi-IN-MadhurNeural", "Madhur", "male", "hi", "hi-IN", "Hindi Male Neural"),
    },
    "ta": {
        "female": VoiceProfile("ta-IN-PallaviNeural", "Pallavi", "female", "ta", "ta-IN", "Tamil Female Neural"),
        "male": VoiceProfile("ta-IN-ValluvarNeural", "Valluvar", "male", "ta", "ta-IN", "Tamil Male Neural"),
    },
    "en": {
        "female": VoiceProfile("en-IN-NeerjaNeural", "Neerja", "female", "en", "en-IN", "Indian English Female Neural"),
        "male": VoiceProfile("en-IN-PrabhatNeural", "Prabhat", "male", "en", "en-IN", "Indian English Male Neural"),
    },
    "en-us": {
        "female": VoiceProfile("en-US-JennyNeural", "Jenny", "female", "en", "en-US", "US English Female Neural"),
        "male": VoiceProfile("en-US-GuyNeural", "Guy", "male", "en", "en-US", "US English Male Neural"),
    },
    "kn": {
        "female": VoiceProfile("kn-IN-SapnaNeural", "Sapna", "female", "kn", "kn-IN", "Kannada Female Neural"),
        "male": VoiceProfile("kn-IN-GaganNeural", "Gagan", "male", "kn", "kn-IN", "Kannada Male Neural"),
    },
    "ml": {
        "female": VoiceProfile("ml-IN-SobhanaNeural", "Sobhana", "female", "ml", "ml-IN", "Malayalam Female Neural"),
        "male": VoiceProfile("ml-IN-MidhunNeural", "Midhun", "male", "ml", "ml-IN", "Malayalam Male Neural"),
    },
    "mr": {
        "female": VoiceProfile("mr-IN-AarohiNeural", "Aarohi", "female", "mr", "mr-IN", "Marathi Female Neural"),
        "male": VoiceProfile("mr-IN-ManoharNeural", "Manohar", "male", "mr", "mr-IN", "Marathi Male Neural"),
    },
    "bn": {
        "female": VoiceProfile("bn-IN-TanishaaNeural", "Tanishaa", "female", "bn", "bn-IN", "Bengali Female Neural"),
        "male": VoiceProfile("bn-IN-BashkarNeural", "Bashkar", "male", "bn", "bn-IN", "Bengali Male Neural"),
    },
    "gu": {
        "female": VoiceProfile("gu-IN-DhwaniNeural", "Dhwani", "female", "gu", "gu-IN", "Gujarati Female Neural"),
        "male": VoiceProfile("gu-IN-NiranjanNeural", "Niranjan", "male", "gu", "gu-IN", "Gujarati Male Neural"),
    },
    "pa": {
        "female": VoiceProfile("pa-IN-OjasNeural", "Ojas", "female", "pa", "pa-IN", "Punjabi Female Neural"),
        "male": VoiceProfile("pa-IN-GurpreetNeural", "Gurpreet", "male", "pa", "pa-IN", "Punjabi Male Neural"),
    },
    "ur": {
        "female": VoiceProfile("ur-IN-GulNeural", "Gul", "female", "ur", "ur-IN", "Urdu Female Neural"),
        "male": VoiceProfile("ur-IN-SalmanNeural", "Salman", "male", "ur", "ur-IN", "Urdu Male Neural"),
    },
    # International Languages
    "es": {
        "female": VoiceProfile("es-ES-ElviraNeural", "Elvira", "female", "es", "es-ES", "Spanish Female Neural"),
        "male": VoiceProfile("es-ES-AlvaroNeural", "Alvaro", "male", "es", "es-ES", "Spanish Male Neural"),
    },
    "fr": {
        "female": VoiceProfile("fr-FR-DeniseNeural", "Denise", "female", "fr", "fr-FR", "French Female Neural"),
        "male": VoiceProfile("fr-FR-HenriNeural", "Henri", "male", "fr", "fr-FR", "French Male Neural"),
    },
    "de": {
        "female": VoiceProfile("de-DE-KatjaNeural", "Katja", "female", "de", "de-DE", "German Female Neural"),
        "male": VoiceProfile("de-DE-ConradNeural", "Conrad", "male", "de", "de-DE", "German Male Neural"),
    },
    "ja": {
        "female": VoiceProfile("ja-JP-NanamiNeural", "Nanami", "female", "ja", "ja-JP", "Japanese Female Neural"),
        "male": VoiceProfile("ja-JP-KeitaNeural", "Keita", "male", "ja", "ja-JP", "Japanese Male Neural"),
    },
    "zh": {
        "female": VoiceProfile("zh-CN-XiaoxiaoNeural", "Xiaoxiao", "female", "zh", "zh-CN", "Mandarin Female Neural"),
        "male": VoiceProfile("zh-CN-YunxiNeural", "Yunxi", "male", "zh", "zh-CN", "Mandarin Male Neural"),
    },
    "ar": {
        "female": VoiceProfile("ar-SA-ZariyahNeural", "Zariyah", "female", "ar", "ar-SA", "Arabic Female Neural"),
        "male": VoiceProfile("ar-SA-HamedNeural", "Hamed", "male", "ar", "ar-SA", "Arabic Male Neural"),
    },
}


class StreamingTtsService:
    """
    Zero-disk-write Streaming TTS Engine.
    Leverages Microsoft Edge neural voices for high-fidelity speech synthesis
    across Indian and international languages.
    """

    def __init__(self, cache_size: int = 128):
        self.voice_registry = DEFAULT_VOICE_REGISTRY
        self.cache_size = cache_size
        self._cache: OrderedDict[Tuple[str, str, str, str, str], SpeechSynthesisResult] = OrderedDict()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_worker")

    def get_voice(self, language_code: str, gender: str = "female") -> VoiceProfile:
        """Resolve the optimal neural voice for the specified language and gender."""
        lang = language_code.lower().strip()
        g = gender.lower().strip()
        if g not in ("male", "female"):
            g = "female"

        # Direct match (e.g. 'te', 'hi', 'en')
        if lang in self.voice_registry and g in self.voice_registry[lang]:
            return self.voice_registry[lang][g]

        # Region-qualified match (e.g. 'en-us', 'en-in')
        base_lang = lang.split("-")[0]
        if base_lang in self.voice_registry and g in self.voice_registry[base_lang]:
            return self.voice_registry[base_lang][g]

        # Fallback to English
        logger.warning(f"Voice not found for lang='{language_code}' gender='{gender}', falling back to English.")
        return self.voice_registry["en"][g]

    async def synthesize_stream_async(self, request: SpeechSynthesisRequest) -> AsyncIterator[bytes]:
        """
        Asynchronously stream raw audio byte packets from Edge-TTS.
        Enables ultra-low Time-To-First-Audio (TTFA) by streaming chunks as they are generated.
        Zero disk writes.
        """
        voice = request.voice_id or self.get_voice(request.language_code, request.gender).voice_id
        communicate = edge_tts.Communicate(
            text=request.text,
            voice=voice,
            rate=request.rate,
            pitch=request.pitch,
            volume=request.volume
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize_async(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResult:
        """
        Synthesize speech directly into memory as a NumPy float32 array.
        Zero disk writes. Decodes MP3 stream in-memory using PySoundFile.
        """
        cache_key = (request.text.strip(), request.language_code, request.gender, request.rate, request.pitch)
        if cache_key in self._cache:
            hit = self._cache[cache_key]
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            return SpeechSynthesisResult(
                audio=hit.audio.copy(),
                sample_rate=hit.sample_rate,
                duration_sec=hit.duration_sec,
                latency_ms=0.1,  # Cache hit
                voice_id=hit.voice_id,
                language_code=hit.language_code,
                num_chunks=hit.num_chunks
            )

        start_time = time.perf_counter()
        voice_id = request.voice_id or self.get_voice(request.language_code, request.gender).voice_id

        communicate = edge_tts.Communicate(
            text=request.text,
            voice=voice_id,
            rate=request.rate,
            pitch=request.pitch,
            volume=request.volume
        )

        bio = io.BytesIO()
        chunk_count = 0

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                bio.write(chunk["data"])
                chunk_count += 1

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if chunk_count == 0 or bio.tell() == 0:
            raise RuntimeError(f"Edge-TTS synthesis returned empty audio stream for text: '{request.text[:30]}...'")

        bio.seek(0)
        audio_data, sample_rate = sf.read(bio, dtype="float32")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        duration_sec = float(len(audio_data)) / float(sample_rate)

        result = SpeechSynthesisResult(
            audio=audio_data,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            latency_ms=latency_ms,
            voice_id=voice_id,
            language_code=request.language_code,
            num_chunks=chunk_count
        )

        # Update LRU cache
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = result

        return result

    def synthesize(
        self,
        text: str,
        language_code: str,
        gender: str = "female",
        voice_id: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ) -> SpeechSynthesisResult:
        """
        Synchronous convenience wrapper for text-to-speech synthesis.
        Safe for use from both synchronous worker threads and asynchronous loops.
        """
        req = SpeechSynthesisRequest(
            text=text,
            language_code=language_code,
            gender=gender,
            voice_id=voice_id,
            rate=rate,
            pitch=pitch,
            volume=volume
        )

        # Check if an asyncio event loop is already active in current thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop (e.g. FastAPI / asyncio thread)
            future = self._thread_pool.submit(asyncio.run, self.synthesize_async(req))
            return future.result()
        else:
            return asyncio.run(self.synthesize_async(req))
