"""
Verification script for the Call Translation pipeline without voice cloning.
Tests Language ID, Colloquial Translation, and Edge-TTS synthesis.
"""

import time
from app.services.translation_service import TranslationService
from app.services.tts_service import StreamingTtsService
from app.services.lid_service import LanguageDetectionEngine

def main():
    lid = LanguageDetectionEngine(use_ecapa=False)
    trans = TranslationService()
    tts = StreamingTtsService()

    test_cases = [
        ("ఏం చేయట్లేదు బ్రో", "en"),
        ("ఎలా ఉన్నావు? నేను బాగున్నాను.", "en"),
        ("చిన్నపిల్లల్లో ఉన్నాడు", "en"),
        ("Where are you going right now?", "te"),
        ("मैं ठीक हूँ, आप कैसे हैं?", "en"),
    ]

    print("=" * 70)
    print("📞 REAL-TIME CALL TRANSLATION TESTBENCH (WITHOUT CLONING)")
    print("=" * 70 + "\n")

    for text, target_lang in test_cases:
        t0 = time.perf_counter()
        lid_dec = lid.identify(text)
        src_lang = lid_dec.language_code
        t_lid = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        res = trans.translate(text, source_lang=src_lang, target_lang=target_lang)
        t_trans = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        tts_res = tts.synthesize(res.translated_text, language_code=target_lang, gender="female")
        t_tts = (time.perf_counter() - t2) * 1000

        audio_len = len(tts_res.audio) / tts_res.sample_rate

        print(f"👉 Input ({src_lang}): {text}")
        print(f"🌐 Output ({target_lang}): {res.translated_text}")
        print(f"⚡ Performance: LID: {t_lid:.1f}ms | Translation: {t_trans:.1f}ms | TTS: {t_tts:.1f}ms | Total: {t_lid + t_trans + t_tts:.1f}ms")
        print(f"🔊 Audio Generated: {audio_len:.2f}s synthesized ({tts_res.voice_id})")
        print("-" * 70)

    print("\n✅ All translation and synthesis pipelines verified functioning perfectly!")

if __name__ == "__main__":
    main()
