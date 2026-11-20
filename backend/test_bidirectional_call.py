"""
Interactive Bidirectional Multilingual Voice Call Simulator with AI Call Partner.
Simulates a real-time call between two speakers across different languages:
Example:
  Uma (English) <== FULL-DUPLEX TRANSLATION CALL ==> Sarah (Telugu)

Turn 1: Uma speaks into microphone: "Hi, how are you?"
  -> System captures audio via realme Buds / default mic
  -> Transcribes: "Hi, how are you"
  -> Translates for Sarah: "హలో, మీరు ఎలా ఉన్నారు?"
  -> Sarah (AI Call Partner) contextually responds: "నేను బాగున్నాను! మీరు ఎలా ఉన్నారు?"
  -> Translated for Uma: "Hey, I'm good! What about you?"
  -> Speaks Sarah's reply aloud through your earbuds/speakers in natural neural voice!

Zero disk writes. Real-time VAD, LID, ASR, NMT, TTS, and acoustic feedback protection.
"""

import sys
import time
import argparse
import numpy as np
import sounddevice as sd

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer
from app.services.lid_service import LanguageDetectionEngine, DynamicLanguageTracker
from app.services.asr_service import StreamingAsrService
from app.services.translation_service import TranslationService
from app.services.tts_service import StreamingTtsService
from app.services.voice_cloner import VoiceCloningService
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code


_FALLBACK_INDEX = 0

def generate_call_partner_reply(user_text: str, reply_lang: str = "en") -> str:
    """
    Intelligent conversational call partner response generator.
    Replies contextually to greetings, typos, food/dinner queries, Telugu/Hindi/Tamil
    phrases in native script or Romanized script (Tenglish/Hinglish), and chit-chat.
    """
    global _FALLBACK_INDEX
    text_lower = user_text.lower().strip()

    # 1. Greetings & "How are you" (tolerates typos like 'how are ou', 'how r u', 'ela unnaru', etc.)
    how_are_you_patterns = [
        "how are you", "how are ou", "how r u", "how u", "hw r u", "how do you do",
        "how are things", "whats up", "what's up", "wassup", "sup",
        # Telugu (Native & Romanized)
        "బాగున్నారా", "ఎలా ఉన్నారు", "ela unnaru", "ela unnav", "bagunnara", "bagunnava", "kusalama",
        # Hindi
        "हालचाल", "कैसे हो", "kaise ho", "kya haal",
        # Tamil
        "எப்படி இருக்கிறீர்கள்", "eppadi irukkeenga", "eppadi irukinga"
    ]
    if any(q in text_lower for q in how_are_you_patterns):
        replies = {
            "en": "Hey, I'm good! What about you?",
            "te": "హలో, నేను చాలా బాగున్నాను! మీరు ఎలా ఉన్నారు?",
            "ta": "வணக்கம், நான் நன்றாக இருக்கிறேன்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "hi": "नमस्ते, मैं बहुत अच्छा हूँ! आप कैसे हैं?"
        }
        return replies.get(reply_lang, replies["en"])

    # 2. General Greetings (hi, hello, namaste, namaskaram)
    greeting_patterns = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "నమస్కారం", "namaskaram", "namaste", "வணக்கம்", "vanakkam", "नमस्ते"
    ]
    if any(q in text_lower for q in greeting_patterns):
        replies = {
            "en": "Hello! Great to hear from you. How are you doing today?",
            "te": "నమస్కారం! మీతో మాట్లాడటం ఆనందంగా ఉంది. ఎలా ఉన్నారు?",
            "ta": "வணக்கம்! உங்களுடன் பேசுவதில் மகிழ்ச்சி. இன்று எப்படி இருக்கிறீர்கள்?",
            "hi": "नमस्ते! आपसे बात करके अच्छा लगा. आज आप कैसे हैं?"
        }
        return replies.get(reply_lang, replies["en"])

    # 3. Meals, Dinner, Lunch, Food queries ("have your dinner", "had lunch", "tinnara", etc.)
    food_patterns = [
        "dinner", "lunch", "breakfast", "food", "eat", "eaten", "having food",
        "have your dinner", "had your dinner", "have dinner", "had dinner",
        # Telugu
        "తిన్నారా", "భోజనం", "tinnara", "thinnara", "thinnava", "tinnava", "bhojanam", "tiffin",
        # Hindi
        "खाना", "khana", "khaya", "khaa liya",
        # Tamil
        "சாப்பிட்டீர்களா", "சாப்பாடு", "sappittacha", "sappadu"
    ]
    if any(q in text_lower for q in food_patterns):
        replies = {
            "en": "Yes, I had my food already! Thank you for asking. Did you have yours?",
            "te": "అవును, నా భోజనం అయిపోయింది! అడిగినందుకు ధన్యవాదాలు. మీరు తిన్నారా?",
            "ta": "ஆம், நான் சாப்பிட்டுவிட்டேன்! நீங்கள் சாப்பிட்டீர்களா?",
            "hi": "हाँ, मैंने खाना खा लिया! पूछने के लिए शुक्रिया. क्या आपने खाया?"
        }
        return replies.get(reply_lang, replies["en"])

    # 4. Can you hear me / audio check
    hear_patterns = [
        "can you hear me", "hear me", "speaking to you", "am i audible", "sound check",
        "వినిపిస్తుందా", "vinipistunda", "vinipistundha",
        "கேட்கிறதா", "ketkiratha", "kekutha",
        "सुन पा रहे", "awaaz aa rahi", "sunai de raha"
    ]
    if any(q in text_lower for q in hear_patterns):
        replies = {
            "en": "Yes, I can hear you loud and clear! The voice translation is working seamlessly.",
            "te": "అవును, మీ మాటలు చాలా స్పష్టంగా వినిపిస్తున్నాయి! అనువాదం చాలా బాగా పనిచేస్తోంది.",
            "ta": "ஆம், நீங்கள் பேசுவது தெளிவாக கேட்கிறது! மொழிபெயர்ப்பு நன்றாக செயல்படுகிறது.",
            "hi": "हाँ, मैं आपको बिल्कुल साफ़ सुन सकता हूँ! ट्रांसलेशन बहुत अच्छा काम कर रहा है."
        }
        return replies.get(reply_lang, replies["en"])

    # 5. What are you doing / Work / Project queries
    activity_patterns = [
        "what are you doing", "what doing", "what project", "about project", "office", "work", "busy",
        "ఏం చేస్తున్నారు", "em chestunnav", "em chestunnaru",
        "என்ன செய்கிறீர்கள்", "enna panreenga",
        "क्या कर रहे", "kya kar rahe ho"
    ]
    if any(q in text_lower for q in activity_patterns):
        replies = {
            "en": "I'm testing our real-time voice call translation system with you right now!",
            "te": "మనం ప్రస్తుతం రియల్ టైమ్ వాయిస్ కాల్ అనువాద వ్యవస్థను పరీక్షిస్తున్నాము!",
            "ta": "நாம் இப்போது நிகழ்நேர குரல் அழைப்பு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்!",
            "hi": "हम अभी हमारे रीयल-टाइम वॉयस कॉल ट्रांसलेशन सिस्टम का परीक्षण कर रहे हैं!"
        }
        return replies.get(reply_lang, replies["en"])

    # 6. Where are you / Location
    location_patterns = [
        "where are you", "where r u", "where do you live", "location",
        "ఎక్కడ ఉన్నారు", "ekkada unnav", "ekkaduntaru",
        "எங்கே இருக்கிறீர்கள்", "enga irukkeenga",
        "कहाँ हो", "kahan ho"
    ]
    if any(q in text_lower for q in location_patterns):
        replies = {
            "en": "I'm right here with you on the call, ready to chat anytime!",
            "te": "నేను ఇక్కడే మీతో కాల్‌లో ఉన్నాను, మీతో మాట్లాడటానికి సిద్ధంగా ఉన్నాను!",
            "ta": "நான் இதோ உங்கள் அழைப்பில் உங்களுடன் பேச தயாராக இருக்கிறேன்!",
            "hi": "मैं यहीं आपके साथ कॉल पर हूँ, आपसे बात करने के लिए हमेशा तैयार!"
        }
        return replies.get(reply_lang, replies["en"])

    # 7. What is your name / Who are you
    name_patterns = [
        "who are you", "your name", "what is your name", "who r u",
        "మీ పేరు", "evaru meeru", "mee peru enti",
        "உங்கள் பெயர்", "unga peyer enna",
        "आपका नाम", "aapka naam kya hai"
    ]
    if any(q in text_lower for q in name_patterns):
        replies = {
            "en": "I am Sarah, your AI call partner conversing with you across languages.",
            "te": "నేను సారా, మీతో వివిధ భాషల్లో మాట్లాడే మీ AI కాల్ భాగస్వామిని.",
            "ta": "நான் சாரா, உங்களுடன் பல மொழிகளில் பேசும் உங்கள் AI அழைப்பு கூட்டாளி.",
            "hi": "मैं सारा हूँ, आपकी AI कॉल पार्टनर जो आपसे विभिन्न भाषाओं में बात करती हूँ."
        }
        return replies.get(reply_lang, replies["en"])

    # 8. Good / Nice / Thank you
    praise_patterns = [
        "thank", "thanks", "good", "great", "awesome", "nice", "super", "perfect",
        "ధన్యవాదాలు", "dhanyavadalu", "chala bagundi",
        "நன்றி", "nandri", "romba nalla irukku",
        "धन्यवाद", "shukriya", "bahut achha"
    ]
    if any(q in text_lower for q in praise_patterns):
        replies = {
            "en": "You're very welcome! It's really exciting to see our speech translation working so smoothly.",
            "te": "చాలా సంతోషం! మన వాయిస్ ట్రాన్స్‌లేషన్ ఇంత అద్భుతంగా పనిచేయడం చాలా బాగుంది.",
            "ta": "மிக்க மகிழ்ச்சி! நமது குரல் மொழிபெயர்ப்பு மிகவும் அற்புதமாக செயல்படுகிறது.",
            "hi": "बहुत-बहुत धन्यवाद! हमारा वॉइस ट्रांसलेशन इतनी अच्छी तरह काम कर रहा है, यह बहुत बढ़िया है."
        }
        return replies.get(reply_lang, replies["en"])

    # 9. Goodbyes
    bye_patterns = [
        "bye", "goodbye", "see you", "talk to you later", "good night", "tata",
        "టాటా", "వెళ్ళొస్తాను", "vellostanu",
        "போய் வருகிறேன்", "poyitu varen",
        "अलविदा", "shubh ratri", "phir milenge"
    ]
    if any(q in text_lower for q in bye_patterns):
        replies = {
            "en": "Goodbye! It was wonderful speaking with you. Have a great day ahead!",
            "te": "సరే ఉంటాను! మీతో మాట్లాడటం చాలా ఆనందంగా ఉంది. మంచి రోజు కావాలని కోరుకుంటున్నాను!",
            "ta": "போய் வருகிறேன்! உங்களுடன் பேசியதில் மகிழ்ச்சி. நல்ல நாளாக அமையட்டும்!",
            "hi": "अलविदा! आपसे बात करके बहुत ख़ुशी हुई. आपका दिन शुभ हो!"
        }
        return replies.get(reply_lang, replies["en"])

    # 10. Natural Dynamic Fallback (cycles through varied realistic replies)
    fallback_pool = {
        "en": [
            "That's really interesting! Tell me more about that.",
            "Got it! How is everything else going with you today?",
            "That sounds great! I completely agree with you.",
            "I hear you! It's really wonderful chatting with you like this.",
            "Nice! What else is on your mind?"
        ],
        "te": [
            "చాలా ఆసక్తికరంగా ఉంది! దీని గురించి మరికొంత చెప్పండి.",
            "అర్థమైంది! ఈ రోజు మీ పనులన్నీ ఎలా సాగుతున్నాయి?",
            "అవును, అది నిజమే! నేను మీతో పూర్తిగా ఏకీభవిస్తున్నాను.",
            "మీతో ఇలా మాట్లాడటం చాలా సంతోషంగా ఉంది.",
            "బాగుంది! ఇంకా ఏమైనా విశేషాలు ఉన్నాయా?"
        ],
        "ta": [
            "மிகவும் சுவாரஸ்யமாக இருக்கிறது! இதைப் பற்றி மேலும் சொல்லுங்கள்.",
            "புரிந்தது! இன்று மற்ற வேலைகள் எப்படி போகின்றன?",
            "நிச்சயமாக, அது புரிகிறது! நான் உங்களுடன் உடன்படுகிறேன்.",
            "உங்களுடன் இப்படி பேசுவது மகிழ்ச்சியாக இருக்கிறது."
        ],
        "hi": [
            "यह बहुत दिलचस्प है! इसके बारे में और बताइए.",
            "समझ गया! आज का बाकी दिन आपका कैसा चल रहा है?",
            "बिल्कुल सही! मैं आपकी बात से पूरी तरह सहमत हूँ.",
            "आपसे इस तरह बात करके बहुत अच्छा लग रहा है."
        ]
    }
    lang_pool = fallback_pool.get(reply_lang, fallback_pool["en"])
    reply = lang_pool[_FALLBACK_INDEX % len(lang_pool)]
    _FALLBACK_INDEX += 1
    return reply


def print_banner(a_name, a_lang, b_name, b_lang, input_device_name, output_device_name):
    print("\n" + "=" * 76)
    print("  BIDIRECTIONAL MULTILINGUAL VOICE CALL (LIVE CONVERSATION PARTNER)")
    print(f"  Party A: {a_name} ({a_lang.upper()})  <== FULL-DUPLEX ==>  Party B: {b_name} ({b_lang.upper()})")
    print("=" * 76)
    print(f"  🎙️ Microphone (Input)     : {input_device_name}")
    print(f"  🔊 Sound Playback (Output) : {output_device_name}")
    print(f"  🔒 Feedback Suppression   : ACTIVE (Microphone auto-mutes during AI speech)")
    print(f"  • When you speak as {a_name}:")
    print(f"    1. System transcribes what you said.")
    print(f"    2. Translates it for {b_name} into {b_lang.upper()}.")
    print(f"    3. {b_name} replies contextually in {b_lang.upper()}.")
    print(f"    4. Translated back to {a_lang.upper()} and SPOKEN ALOUD to you!")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Bidirectional Two-Speaker Call Simulator")
    parser.add_argument("--party-a", type=str, default="en", help="Speaker A language (default: en / English)")
    parser.add_argument("--party-b", type=str, default="te", help="Speaker B language (default: te / Telugu)")
    parser.add_argument("--name-a", type=str, default="Uma", help="Speaker A name (default: Uma)")
    parser.add_argument("--name-b", type=str, default="Sarah", help="Speaker B name (default: Sarah)")
    parser.add_argument("--gender-a", type=str, default="female", choices=["female", "male"], help="Speaker A voice gender")
    parser.add_argument("--gender-b", type=str, default="female", choices=["female", "male"], help="Speaker B voice gender")
    parser.add_argument("--device", type=int, default=None, help="Microphone input device index (default: auto-detected)")
    parser.add_argument("--out-device", type=int, default=None, help="Audio output device index (default: auto-detected matching output)")
    parser.add_argument("--speaker", action="store_true", help="Force sound output through laptop physical speakers instead of earphones")
    parser.add_argument("--duration", type=float, default=None, help="Call duration limit in seconds (default: runs indefinitely until Ctrl+C)")
    parser.add_argument("--text", action="store_true", help="Interactive keyboard text mode (type instead of speaking)")
    args = parser.parse_args()

    lang_a = normalize_language_code(args.party_a, default="en")
    lang_b = normalize_language_code(args.party_b, default="te")

    info_a = SUPPORTED_LANGUAGES.get(lang_a, LanguageInfo(lang_a, lang_a.title(), lang_a))
    info_b = SUPPORTED_LANGUAGES.get(lang_b, LanguageInfo(lang_b, lang_b.title(), lang_b))

    name_a = args.name_a
    name_b = args.name_b

    # 1. Initialize Engines
    print("Loading AI pipeline (VoxLingua107 ECAPA LID, Faster-Whisper INT8, Neural NMT, Neural TTS)...")
    from faster_whisper import WhisperModel
    shared_whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
    LanguageDetectionEngine.get_ecapa_model()

    lid_engine = LanguageDetectionEngine(whisper_model=shared_whisper, model_size="base")
    asr_service = StreamingAsrService(model_size="base", shared_model=shared_whisper)
    translation = TranslationService()
    tts = StreamingTtsService()

    streamer = LiveAudioStreamer(
        device_index=args.device,
        vad_start_threshold=0.35,
        vad_end_threshold=0.25,
        hangover_ms=1000,
        min_speech_ms=300,
        max_speech_ms=9000
    )

    # Determine optimal output device
    if args.speaker:
        out_idx = AudioPlaybackEngine.get_laptop_speakers()
    elif args.out_device is not None:
        out_idx = args.out_device
    else:
        out_idx = AudioPlaybackEngine.auto_detect_output_device(streamer.device_index)

    player = AudioPlaybackEngine(device_index=out_idx)

    # Resolve active device names for UI display
    in_name = "Default Microphone"
    out_name = "Default Speakers"
    try:
        active_in = streamer.device_index if streamer.device_index is not None else sd.default.device[0]
        in_info = sd.query_devices(active_in)
        in_name = f"[{active_in}] {in_info.get('name')}"
    except Exception:
        pass
    try:
        active_out = player.device_index if player.device_index is not None else sd.default.device[1]
        out_info = sd.query_devices(active_out)
        out_name = f"[{active_out}] {out_info.get('name')}"
    except Exception:
        pass

    print_banner(name_a, lang_a, name_b, lang_b, in_name, out_name)

    # Echo feedback prevention
    link_playback_to_streamer(player, streamer, duck_only=False)

    # Startup gentle chime to confirm playback is active
    try:
        t_arr = np.linspace(0, 0.35, int(24000 * 0.35), endpoint=False)
        chime = (0.20 * np.sin(2 * np.pi * 523.25 * t_arr) * np.exp(-t_arr * 6)).astype(np.float32)
        player.play(chime, sample_rate=24000, blocking=True)
    except Exception:
        pass

    turn_count = 0

    def process_turn(audio_samples: np.ndarray, sample_rate: int = 16000):
        nonlocal turn_count
        turn_count += 1

        t0 = time.perf_counter()

        # Software AGC (Peak Normalization) for Bluetooth headsets / quiet mics
        peak = float(np.max(np.abs(audio_samples))) if len(audio_samples) > 0 else 0.0
        if peak > 0.003:
            audio_samples = (audio_samples / peak * 0.90).astype(np.float32)

        # Step 1: Speech-to-Text ASR
        t_asr = time.perf_counter()
        asr_res = asr_service.transcribe(audio_or_path=audio_samples, language=lang_a, beam_size=1)
        asr_ms = (time.perf_counter() - t_asr) * 1000.0

        spoken_text = asr_res.text.strip()
        clean_words = "".join(c for c in spoken_text if c.isalnum())
        if not clean_words or len(clean_words) < 2:
            return

        # Step 2: Translation for Party B
        t_trans = time.perf_counter()
        if lang_a == lang_b:
            trans_for_b = spoken_text
            trans_ms = 0.0
        else:
            trans_res = translation.translate(
                text=spoken_text,
                source_lang=lang_a,
                target_lang=lang_b,
                speaker_name=name_a,
                receiver_name=name_b
            )
            trans_for_b = trans_res.translated_text
            trans_ms = (time.perf_counter() - t_trans) * 1000.0

        print("\n" + "=" * 76)
        print(f"📞 [CALL TURN #{turn_count}]")
        print(f"   👤 {name_a} Spoke ({info_a.name} [{lang_a.upper()}]):")
        print(f"      🗣️  \"{spoken_text}\" (ASR Latency: {asr_ms:.0f}ms)")
        if lang_a != lang_b:
            print(f"   🌐 Delivered to {name_b} ({info_b.name} [{lang_b.upper()}]):")
            print(f"      📝 \"{trans_for_b}\"")
        print("   " + "-" * 72)

        # Step 3: Call Partner (Sarah) Contextual Response
        reply_in_b_lang = generate_call_partner_reply(spoken_text, reply_lang=lang_b)

        # Translate reply to A's language (if different)
        if lang_a == lang_b:
            spoken_reply_for_a = reply_in_b_lang
        else:
            reply_trans = translation.translate(
                text=reply_in_b_lang,
                source_lang=lang_b,
                target_lang=lang_a,
                speaker_name=name_b,
                receiver_name=name_a
            )
            spoken_reply_for_a = reply_trans.translated_text

        print(f"   👤 {name_b}'s Reply ({info_b.name} [{lang_b.upper()}]):")
        print(f"      🗣️  \"{reply_in_b_lang}\"")
        if lang_a != lang_b:
            print(f"   🌐 Delivered to {name_a} ({info_a.name} [{lang_a.upper()}]):")
            print(f"      📝 \"{spoken_reply_for_a}\"")

        # Step 4: Synthesize Neural Voice and Speak Aloud
        try:
            tts_lang = lang_a if lang_a == lang_b else lang_a
            tts_res = tts.synthesize(
                text=spoken_reply_for_a,
                language_code=tts_lang,
                gender=args.gender_b
            )
            print(f"   🎙️ Neural Voice: {tts_res.voice_id}")
            print(f"   🔊 Speaking {name_b}'s reply aloud through your earphones/speakers...")
            print("=" * 76)
            sys.stdout.flush()
            player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)
            print(f"\n[Microphone Active] Listening for {name_a} to speak again...")
            sys.stdout.flush()
        except Exception as e:
            print(f"   ❌ TTS Playback error: {e}")

    # Interactive Keyboard Text Mode
    if args.text:
        print("⌨️  INTERACTIVE KEYBOARD MODE")
        print(f"   Type a sentence as {name_a} and press Enter.")
        print(f"   {name_b} will reply contextually and speak back aloud!")
        print("   Type 'exit' to quit.\n" + "-" * 76)

        while True:
            try:
                user_msg = input(f"\n[{name_a} - {lang_a.upper()}] > ").strip()
                if not user_msg or user_msg.lower() in ("exit", "quit", "q"):
                    break

                turn_count += 1
                # Translation for B
                if lang_a == lang_b:
                    trans_for_b = user_msg
                else:
                    tr = translation.translate(user_msg, source_lang=lang_a, target_lang=lang_b)
                    trans_for_b = tr.translated_text

                print(f"   🌐 Delivered to {name_b}: \"{trans_for_b}\"")

                # B's contextual reply
                reply_in_b = generate_call_partner_reply(user_msg, reply_lang=lang_b)
                if lang_a == lang_b:
                    reply_for_a = reply_in_b
                else:
                    tr2 = translation.translate(reply_in_b, source_lang=lang_b, target_lang=lang_a)
                    reply_for_a = tr2.translated_text

                print(f"   👤 {name_b} ({lang_b.upper()}): \"{reply_in_b}\"")
                print(f"   🌐 Translated to {lang_a.upper()}: \"{reply_for_a}\"")
                print(f"   🔊 Speaking aloud through your earphones/speakers...")

                tts_res = tts.synthesize(text=reply_for_a, language_code=lang_a, gender=args.gender_b)
                player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)

            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting call session.")
        return

    # Microphone Live Mode
    streamer.register_on_utterance_complete(lambda seg: process_turn(seg.audio, seg.sample_rate))
    streamer.start()

    print("🟢 CALL IN PROGRESS!")
    print(f"👉 Speak into your microphone as {name_a} in {info_a.name} ({lang_a.upper()}).")
    print(f"   Example: \"Hi, how are you?\" or \"Can you hear me?\"")
    print(f"   {name_b} will hear your translated words and reply to you in spoken voice!")
    if args.duration:
        print(f"   Call will run for {args.duration:.0f} seconds (or press Ctrl+C to terminate).")
    else:
        print("   Press Ctrl+C to terminate the call.\n")

    t_start = time.time()
    try:
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        while True:
            if args.duration and (time.time() - t_start >= args.duration):
                print(f"\n\n⏱️ Duration limit ({args.duration:.0f}s) reached.")
                break
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            is_speaking = prob >= 0.35
            tag = "🗣️ SPEAKING" if is_speaking else "👂 LISTENING"
            bar_len = int(max(0, min(14, (rms + 60) / 4)))
            vu_bar = "█" * bar_len + "░" * (14 - bar_len)
            muted = " [MIC MUTED - AI SPEAKING]" if player.is_playing else ""

            sys.stdout.write(f"\r{spinner[idx % 4]} [{tag}] [{vu_bar}] {rms:5.1f} dBFS | VAD: {prob:4.2f}{muted}   ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\n\nTerminating call session...")
    finally:
        streamer.stop()
        player.stop()

    print(f"\n📞 Call session ended. Total turns: {turn_count}\n")


if __name__ == "__main__":
    main()
