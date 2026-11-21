"""
Intelligent Real-Time Translation Service.
Provides bidirectional translation with named entity protection,
SOV clause segmentation, conversational context preservation, and multi-tier routing.
Tier 1: Cloud API (Gemini/Sarvam if configured).
Tier 2a: Google Free Neural NMT (<250ms, colloquial Indian dialogue support).
Tier 2b: MyMemory Neural Translation API.
Tier 3: Offline phrase bank.
"""

import os
import time
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Tuple

from app.models.translation import TranslationResult
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code
from app.services.entity_preserver import EntityPreserver

logger = logging.getLogger("call_translator.translation")

# Incomplete trailing particles in Indian SOV languages.
# Prevents premature translation cuts before the speaker finishes negating or completing the clause.
TRAILING_INCOMPLETE_MARKERS = {
    "te": ("కాదు", "లేదు", "వద్దు", "అయితే", "మరి", "కూడా", "కానీ"),
    "hi": ("नहीं", "मत", "लेकिन", "अगर", "भी", "कि"),
    "ta": ("இல்லை", "வேண்டாம்", "ஆனால்", "கூட"),
    "kn": ("ಇಲ್ಲ", "ಬೇಡ", "ಆದರೆ"),
    "ml": ("ഇല്ല", "വേണ്ട", "പക്ഷേ")
}

# Colloquial Indian conversational idiom bank (maps spoken idioms to natural colloquial English)
COLLOQUIAL_IDIOM_BANK: Dict[Tuple[str, str, str], str] = {
    ("te", "en", "ఏం చేయట్లేదు బ్రో"): "Not doing anything, bro",
    ("te", "en", "ఏం చేయట్లేదు"): "Not doing anything",
    ("te", "en", "ఏమీ చేయట్లేదు"): "Not doing anything",
    ("te", "en", "ఏం లేదు బ్రో"): "Nothing much, bro",
    ("te", "en", "ఏం లేదు"): "Nothing much",
    ("te", "en", "ఏమీ లేదు"): "Nothing",
    ("te", "en", "చిన్నపిల్లల్లో ఉన్నాడు"): "He is with the kids",
    ("te", "en", "పిల్లల్లో ఉన్నాడు"): "He is with the children",
    ("te", "en", "పిల్లలతో ఉన్నాడు"): "He is with the kids",
    ("te", "en", "ఏం మాట్లాడు"): "Say something",
    ("te", "en", "ఏం మాట్లాడాలి"): "What should I say?",
    ("te", "en", "ఏం మాట్లాడుతున్నావు"): "What are you talking about?",
    ("te", "en", "అయితే ఓకే ఎప్పుడు"): "Alright then, when?",
    ("te", "en", "అయితే ఓకే"): "Alright then",
    ("te", "en", "అంతా బాగుంది"): "Everything is good",
    ("te", "en", "నువ్వు ఎలా ఉన్నావు"): "How are you doing?",
    ("te", "en", "ఎలా ఉన్నారు"): "How are you?",
    ("te", "en", "తిన్నావా"): "Had your food?",
    ("te", "en", "భోజనం చేశారా"): "Did you have your meal?",
    ("te", "en", "సరే బ్రో"): "Alright bro",
    ("te", "en", "కలుద్దాం"): "Let's meet up",
    ("te", "en", "రేపు మాట్లాడుకుందాం"): "Let's talk tomorrow",
}

# Offline common phrase bank for zero-internet testing
OFFLINE_PHRASE_BANK: Dict[Tuple[str, str, str], str] = {
    ("te", "en", "నమస్కారం, మీరు ఎలా ఉన్నారు?"): "Hello, how are you?",
    ("en", "te", "Hello, how are you?"): "నమస్కారం, మీరు ఎలా ఉన్నారు?",
    ("en", "te", "Hello, how are you today?"): "నమస్కారం, మీరు ఈరోజు ఎలా ఉన్నారు?",
    ("te", "en", "నేను రేపు ఉదయం కాల్ చేస్తాను."): "I will call tomorrow morning.",
    ("en", "te", "I will call tomorrow morning."): "నేను రేపు ఉదయం కాల్ చేస్తాను.",
    ("hi", "en", "नमस्ते, आप कैसे हैं?"): "Hello, how are you?",
    ("en", "hi", "Hello, how are you?"): "नमस्ते, आप कैसे हैं?",
    ("ta", "en", "வணக்கம், நீங்கள் நலமா?"): "Hello, how are you?",
    ("en", "ta", "Hello, how are you?"): "வணக்கம், நீங்கள் நலமா?",
    ("te", "hi", "నమస్కారం, మీరు ఎలా ఉన్నారు?"): "नमस्ते, आप कैसे हैं?",
    ("hi", "te", "नमस्ते, आप कैसे हैं?"): "నమస్కారం, మీరు ఎలా ఉన్నారు?",
}


class ClauseSegmentationBuffer:
    """
    Buffers streaming ASR tokens until a safe syntactic clause boundary is reached.
    Prevents splitting trailing negation particles in Indian SOV languages.
    """

    def __init__(self, max_wait_tokens: int = 14):
        self.max_wait_tokens = max_wait_tokens
        self._tokens: List[str] = []

    def push(self, text: str, lang: str = "te", force_flush: bool = False) -> List[str]:
        words = text.strip().split()
        if not words:
            return []

        self._tokens.extend(words)
        joined = " ".join(self._tokens)

        # Check for clause terminal punctuation
        ends_punct = joined.endswith((".", "?", "!", "।", "॥", ","))

        # Check if trailing word is an incomplete negation particle
        lang_markers = TRAILING_INCOMPLETE_MARKERS.get(lang, ())
        ends_incomplete = any(joined.endswith(m) for m in lang_markers)

        if force_flush or (ends_punct and not ends_incomplete) or len(self._tokens) >= self.max_wait_tokens:
            self._tokens.clear()
            return [joined]

        return []

    def clear(self) -> None:
        self._tokens.clear()


class TranslationService:
    """
    Multi-Tier Translation Engine.
    Tier 1: Cloud API (Gemini / Sarvam) if configured in environment.
    Tier 2a: Google Free Neural NMT (Zero API keys, fast <250ms).
    Tier 2b: MyMemory Neural Translation API.
    Tier 3: Offline Phrase Cache & Fallback.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, sarvam_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.sarvam_api_key = sarvam_api_key or os.environ.get("SARVAM_API_KEY")
        self.entity_preserver = EntityPreserver()
        self._cache: Dict[Tuple[str, str, str], str] = {}
        self.clause_buffer = ClauseSegmentationBuffer()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context_history: Optional[List[str]] = None,
        speaker_name: Optional[str] = None,
        receiver_name: Optional[str] = None,
        preserve_entities: bool = True
    ) -> TranslationResult:
        """
        Translate text from source_lang to target_lang with entity protection.
        Automatically canonicalizes language names ('hindi', 'eng', 'telugu') to 2-letter ISO codes.
        """
        t0 = time.perf_counter()
        clean = " ".join(text.strip().split())

        # Canonicalize language codes (e.g. 'hindi' -> 'hi', 'eng' -> 'en')
        src_iso = normalize_language_code(source_lang, default="te")
        tgt_iso = normalize_language_code(target_lang, default="en")

        src_info = SUPPORTED_LANGUAGES.get(src_iso, LanguageInfo(src_iso, src_iso.title(), src_iso))
        tgt_info = SUPPORTED_LANGUAGES.get(tgt_iso, LanguageInfo(tgt_iso, tgt_iso.title(), tgt_iso))

        # 1. Identity Check
        if not clean or src_iso == tgt_iso:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return TranslationResult(
                source_text=clean,
                translated_text=clean,
                source_language=src_iso,
                target_language=tgt_iso,
                source_language_name=src_info.name,
                target_language_name=tgt_info.name,
                backend="identity_passthrough",
                latency_ms=elapsed,
                preserved_entities=[]
            )

        # 2. Check Colloquial Idiom Bank for conversational idioms
        clean_no_punct = clean.strip(".?,! ")
        idiom_match = (
            COLLOQUIAL_IDIOM_BANK.get((src_iso, tgt_iso, clean)) or
            COLLOQUIAL_IDIOM_BANK.get((src_iso, tgt_iso, clean_no_punct))
        )
        if idiom_match:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return TranslationResult(
                source_text=clean,
                translated_text=idiom_match,
                source_language=src_iso,
                target_language=tgt_iso,
                source_language_name=src_info.name,
                target_language_name=tgt_info.name,
                backend="colloquial_idiom_bank",
                latency_ms=elapsed,
                preserved_entities=[]
            )

        # 3. Check in-memory cache
        cache_key = (clean, src_iso, tgt_iso)
        if cache_key in self._cache:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return TranslationResult(
                source_text=clean,
                translated_text=self._cache[cache_key],
                source_language=src_iso,
                target_language=tgt_iso,
                source_language_name=src_info.name,
                target_language_name=tgt_info.name,
                backend="memory_cache",
                latency_ms=elapsed,
                preserved_entities=[]
            )

        # Exact offline phrases are deterministic, private, and faster than a network call.
        offline_match = OFFLINE_PHRASE_BANK.get((src_iso, tgt_iso, clean))
        if offline_match is not None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._cache[cache_key] = offline_match
            return TranslationResult(
                source_text=clean,
                translated_text=offline_match,
                source_language=src_iso,
                target_language=tgt_iso,
                source_language_name=src_info.name,
                target_language_name=tgt_info.name,
                backend="offline_phrase_bank",
                latency_ms=elapsed,
                preserved_entities=[],
            )

        # 3. Named Entity & Technical Term Protection
        masked_text = clean
        placeholder_map: Dict[str, str] = {}
        names_to_protect = []
        if speaker_name:
            names_to_protect.append(speaker_name)
        if receiver_name:
            names_to_protect.append(receiver_name)

        if preserve_entities:
            masked_text, placeholder_map = self.entity_preserver.mask(clean, additional_names=names_to_protect)

        # 4. Multi-Tier Routing & Translation Execution
        backend_used = "offline_phrase_bank"
        translated_masked = ""

        # Tier 1: Gemini API (if key present)
        if self.gemini_api_key:
            try:
                translated_masked = self._translate_gemini(
                    masked_text, src_info.name, tgt_info.name, context_history
                )
                backend_used = "gemini_flash_nmt"
            except Exception as e:
                logger.warning(f"Gemini translation failed ({e}), falling back to Tier 2 Free NMT...")

        # Tier 2a: Google Free Neural NMT Engine (ultra-fast, supports colloquial Indian sentences)
        if not translated_masked:
            try:
                translated_masked = self._translate_google_free(masked_text, src_iso, tgt_iso)
                backend_used = "google_neural_nmt"
            except Exception as e_google:
                logger.debug(f"Google free NMT error ({e_google}), trying MyMemory...")

        # Tier 2b: MyMemory API Fallback
        if not translated_masked:
            try:
                translated_masked = self._translate_mymemory(masked_text, src_iso, tgt_iso)
                backend_used = "mymemory_neural_nmt"
            except Exception as e_mymemory:
                logger.warning(f"MyMemory free NMT error ({e_mymemory}), checking offline phrase bank...")

        # Tier 3: Offline phrase bank fallback
        if not translated_masked:
            translated_masked = OFFLINE_PHRASE_BANK.get(
                (src_iso, tgt_iso, clean),
                clean  # Graceful fallback to original text
            )
            backend_used = "offline_phrase_bank"

        # 5. Entity Unmasking
        if preserve_entities and placeholder_map:
            final_text = self.entity_preserver.unmask(translated_masked, placeholder_map)
        else:
            final_text = translated_masked

        # Clean punctuation spacing
        final_text = " ".join(final_text.strip().split())

        # Conversational Indian NMT Refinement (recovers from literal Google Translate parse errors)
        if src_iso == "te" and tgt_iso == "en":
            low_final = final_text.lower().strip()
            if "what can you do" in low_final and ("చేయట్లేదు" in clean or "చెయ్యట్లేదు" in clean):
                final_text = re.sub(r"(?i)bro,?\s*what can you do\??", "Not doing anything, bro", final_text)
                final_text = re.sub(r"(?i)what can you do\??", "Not doing anything", final_text)
            elif "in childhood" in low_final and "పిల్లల్లో" in clean:
                final_text = re.sub(r"(?i)he is in childhood", "He is with the kids", final_text)
                final_text = re.sub(r"(?i)in childhood", "with the children", final_text)
            elif low_final in ("what talk", "what talk?") and "మాట్లాడు" in clean:
                final_text = "Say something"

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Update cache (keep max 100 entries)
        if len(self._cache) > 100:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = final_text

        return TranslationResult(
            source_text=clean,
            translated_text=final_text,
            source_language=src_iso,
            target_language=tgt_iso,
            source_language_name=src_info.name,
            target_language_name=tgt_info.name,
            backend=backend_used,
            latency_ms=elapsed,
            preserved_entities=list(placeholder_map.values())
        )

    def _translate_google_free(self, text: str, src: str, tgt: str) -> str:
        """
        Ultra-fast public neural translation via Google Translate single client.
        High quality, fast latency (<250ms), supports Indian colloquial idioms and dialects.
        Zero API keys required.
        """
        encoded_query = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={encoded_query}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "*/*"
            }
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                translated_parts = [part[0] for part in data[0] if part and part[0]]
                translated = "".join(translated_parts).strip()
                if translated:
                    return translated
        raise RuntimeError("No translation returned by Google Translate")

    def _translate_mymemory(self, text: str, src: str, tgt: str) -> str:
        """
        Public neural translation via MyMemory API.
        """
        encoded_query = urllib.parse.quote(text)
        url = f"https://api.mymemory.translated.net/get?q={encoded_query}&langpair={src}|{tgt}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "MultilingualVoiceAI/1.0",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated = data.get("responseData", {}).get("translatedText")
            if translated and translated.strip():
                return translated.strip()

        raise RuntimeError("No translation returned by MyMemory")

    def _translate_gemini(
        self,
        text: str,
        src_name: str,
        tgt_name: str,
        context_history: Optional[List[str]] = None
    ) -> str:
        """
        Cloud neural LLM translation via Google Gemini 1.5 Flash (if API key configured).
        """
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        system_instruction = (
            f"You are a professional real-time voice call interpreter. "
            f"Translate the following spoken conversational utterance from {src_name} to {tgt_name}.\n"
            f"Preserve all tokens enclosed in <entity_...> tags verbatim. "
            f"Do not add commentary, explanations, or notes. Output ONLY the translated sentence."
        )

        prompt = f"{system_instruction}\n\nUtterance: {text}"
        if context_history:
            recent_ctx = "\n".join(context_history[-3:])
            prompt = f"{system_instruction}\n\nRecent Conversation Context:\n{recent_ctx}\n\nUtterance to translate: {text}"

        response = model.generate_content(prompt)
        if response.text:
            return response.text.strip()
        raise RuntimeError("Empty response from Gemini API")


class BidirectionalCallTranslator:
    """
    Coordinates symmetric two-party translation session.
    Manages speaker A <-> speaker B turns and synchronizes contextual history.
    """

    def __init__(
        self,
        translation_service: Optional[TranslationService] = None,
        party_a_lang: str = "te",
        party_b_lang: str = "en",
        party_a_name: str = "Caller A",
        party_b_name: str = "Caller B"
    ):
        self.service = translation_service or TranslationService()
        self.party_a_lang = party_a_lang
        self.party_b_lang = party_b_lang
        self.party_a_name = party_a_name
        self.party_b_name = party_b_name
        self.conversation_history: List[str] = []

    def translate_a_to_b(self, text: str) -> TranslationResult:
        res = self.service.translate(
            text,
            source_lang=self.party_a_lang,
            target_lang=self.party_b_lang,
            speaker_name=self.party_a_name,
            receiver_name=self.party_b_name,
            context_history=self.conversation_history
        )
        self.conversation_history.append(f"{self.party_a_name} ({self.party_a_lang}): {text}")
        self.conversation_history.append(f"{self.party_b_name} ({self.party_b_lang}): {res.translated_text}")
        return res

    def translate_b_to_a(self, text: str) -> TranslationResult:
        res = self.service.translate(
            text,
            source_lang=self.party_b_lang,
            target_lang=self.party_a_lang,
            speaker_name=self.party_b_name,
            receiver_name=self.party_a_name,
            context_history=self.conversation_history
        )
        self.conversation_history.append(f"{self.party_b_name} ({self.party_b_lang}): {text}")
        self.conversation_history.append(f"{self.party_a_name} ({self.party_a_lang}): {res.translated_text}")
        return res

    def translate_caller_a_to_b(self, text: str, lang_a: Optional[str] = None, lang_b: Optional[str] = None) -> TranslationResult:
        if lang_a:
            self.party_a_lang = lang_a
        if lang_b:
            self.party_b_lang = lang_b
        return self.translate_a_to_b(text)

    def translate_caller_b_to_a(self, text: str, lang_b: Optional[str] = None, lang_a: Optional[str] = None) -> TranslationResult:
        if lang_b:
            self.party_b_lang = lang_b
        if lang_a:
            self.party_a_lang = lang_a
        return self.translate_b_to_a(text)

    def clear(self):
        self.conversation_history.clear()
        self.service.clause_buffer.clear()
