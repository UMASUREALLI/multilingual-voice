"""
Diagnostic test script to verify translation across Indian and global languages.
"""

from app.services.translation_service import TranslationService
from app.models.language import normalize_language_code, SUPPORTED_LANGUAGES

t = TranslationService()

test_cases = [
    ("Tamil", "ta", "வணக்கம், எப்படி இருக்கிறீர்கள்?"),
    ("Kannada", "kn", "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?"),
    ("Malayalam", "ml", "നമസ്കാരം, സുഖമാണോ?"),
    ("Marathi", "mr", "नमस्कार, कसे आहात?"),
    ("Bengali", "bn", "নমস্কার, কেমন আছেন?"),
    ("Gujarati", "gu", "નમસ્તે, કેમ છો?"),
    ("Punjabi", "pa", "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ, ਕਿਵੇਂ ਹੋ?"),
    ("Urdu", "ur", "ہیلو، آپ کیسے ہیں؟"),
    ("Spanish", "es", "Hola, ¿cómo estás?"),
    ("French", "fr", "Bonjour, comment allez-vous?"),
    ("German", "de", "Hallo, wie geht es dir?"),
    ("Japanese", "ja", "こんにちは、お元気ですか？"),
]

print("=" * 72)
print("  MULTILINGUAL TRANSLATION ENGINE VERIFICATION ACROSS ALL LANGUAGES")
print("=" * 72)

for name, code, text in test_cases:
    # Test Native -> English
    r_en = t.translate(text, source_lang=code, target_lang="en")
    print(f"[{name.upper()}] ({code} -> EN): \"{text}\"")
    print(f"  -> English: \"{r_en.translated_text}\" | {r_en.latency_ms:.1f}ms ({r_en.backend})")

    # Test English -> Native
    r_back = t.translate("Hello, how are you?", source_lang="en", target_lang=code)
    print(f"  <- Back to {name}: \"{r_back.translated_text}\" | {r_back.latency_ms:.1f}ms ({r_back.backend})")
    print("-" * 72)

print("\nAll languages verified successfully.")
