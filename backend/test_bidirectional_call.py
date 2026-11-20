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

import os
import sys
import time
import json
import argparse
import urllib.request
from typing import Optional
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


from app.services.llm_dialogue import DynamicDialogueService

_dialogue_service: Optional[DynamicDialogueService] = None


def get_dialogue_service(gemini_key: Optional[str] = None, user_name: str = "Uma", partner_name: str = "Sarah") -> DynamicDialogueService:
    global _dialogue_service
    if _dialogue_service is None:
        _dialogue_service = DynamicDialogueService(
            gemini_api_key=gemini_key,
            user_name=user_name,
            partner_name=partner_name
        )
    return _dialogue_service


def generate_call_partner_reply(user_text: str, reply_lang: str = "en", gemini_key: Optional[str] = None, user_name: str = "Uma", partner_name: str = "Sarah") -> str:
    """
    100% Dynamic Generative AI Call Partner response generator.
    Zero static answers. Zero hardcoded if-else statements.
    Generates open-ended neural responses for any input via local Qwen2.5-0.5B (or Gemini if key provided).
    """
    service = get_dialogue_service(gemini_key=gemini_key, user_name=user_name, partner_name=partner_name)
    return service.generate_reply(user_text, reply_lang=reply_lang)


def print_banner(a_name, a_lang, b_name, b_lang, input_device_name, output_device_name, llm_active: bool = False):
    print("\n" + "=" * 76)
    print("  BIDIRECTIONAL MULTILINGUAL VOICE CALL (LIVE CONVERSATION PARTNER)")
    print(f"  Party A: {a_name} ({a_lang.upper()})  <== FULL-DUPLEX ==>  Party B: {b_name} ({b_lang.upper()})")
    print("=" * 76)
    print(f"  🎙️ Microphone (Input)     : {input_device_name}")
    print(f"  🔊 Sound Playback (Output) : {output_device_name}")
    print(f"  🔒 Feedback Suppression   : ACTIVE (Microphone auto-mutes during AI speech)")
    brain_info = "⚡ Google Gemini LLM (Cloud Generative Intelligence)" if llm_active else "🤖 Qwen2.5-0.5B-Instruct Neural SLM (100% Dynamic On-Device Model)"
    print(f"  🧠 Conversational Brain   : {brain_info}")
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
    parser.add_argument("--gemini-key", type=str, default=None, help="Google Gemini API Key for unlimited real-time AI reasoning to any question")
    parser.add_argument("--demo", action="store_true", help="Play multilingual voice demo speaking in 6 languages")
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

    # Pre-initialize dynamic dialogue service
    dialogue_service = get_dialogue_service(gemini_key=args.gemini_key, user_name=name_a, partner_name=name_b)
    if not (args.gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("🧠 Pre-loading local dynamic dialogue model (Qwen2.5-0.5B-Instruct on CPU)...")
        dialogue_service.load_local_engine()

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

    is_llm = bool(args.gemini_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    print_banner(name_a, lang_a, name_b, lang_b, in_name, out_name, llm_active=is_llm)

    # Echo feedback prevention
    link_playback_to_streamer(player, streamer, duck_only=False)

    # Startup gentle chime to confirm playback is active
    try:
        t_arr = np.linspace(0, 0.35, int(24000 * 0.35), endpoint=False)
        chime = (0.20 * np.sin(2 * np.pi * 523.25 * t_arr) * np.exp(-t_arr * 6)).astype(np.float32)
        player.play(chime, sample_rate=24000, blocking=True)
    except Exception:
        pass

    # Multilingual Voice Showcase Demo
    if args.demo:
        print("\n" + "=" * 76)
        print("🌍 MULTILINGUAL VOICE DEMO: Speaking 6 Languages Through Your Speakers!")
        print("=" * 76)
        demo_phrases = [
            ("en", "Hello Uma! This is our real-time voice translation system speaking in English."),
            ("te", "నమస్కారం ఉమా గారు! మన వాయిస్ సిస్టమ్ తెలుగులో స్పష్టంగా మాట్లాడగలదు."),
            ("hi", "नमस्ते उमा जी! हमारा सिस्टम हिंदी में भी बहुत स्वाभाविक बोलता है."),
            ("ta", "வணக்கம் உமா! நமது அமைப்பு தமிழிலும் மிக அழகாக பேசும்."),
            ("kn", "ನಮಸ್ಕಾರ ಉಮಾ! ನಮ್ಮ ಸಿಸ್ಟಮ್ ಕನ್ನಡದಲ್ಲೂ ಮಾತನಾಡಬಲ್ಲದು."),
            ("es", "¡Hola Uma! Nuestro sistema también habla español con fluidez.")
        ]
        for lang_code, phrase in demo_phrases:
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, LanguageInfo(lang_code, lang_code.title(), lang_code)).name
            print(f"\n🗣️ Speaking in {lang_name} [{lang_code.upper()}]:")
            print(f"   \"{phrase}\"")
            tts_res = tts.synthesize(text=phrase, language_code=lang_code, gender="female")
            print(f"   🎙️ Neural Voice: {tts_res.voice_id} ({tts_res.duration_sec:.1f}s)")
            sys.stdout.flush()
            player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)
            time.sleep(0.4)
        print("\n" + "=" * 76)
        print("✅ MULTILINGUAL DEMO COMPLETE: 6 languages spoken aloud!")
        print("=" * 76 + "\n")
        player.stop()
        return

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
        reply_in_b_raw = generate_call_partner_reply(
            spoken_text,
            reply_lang=lang_b,
            gemini_key=args.gemini_key,
            user_name=name_a,
            partner_name=name_b
        )
        if lang_b == "en":
            reply_in_b_lang = reply_in_b_raw
        else:
            reply_in_b_lang = translation.translate(reply_in_b_raw, source_lang="en", target_lang=lang_b).translated_text

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
            if lang_b != lang_a:
                print(f"   🔊 [1/2] Speaking {name_b}'s reply in {info_b.name} [{lang_b.upper()}]...")
                tts_b = tts.synthesize(text=reply_in_b_lang, language_code=lang_b, gender=args.gender_b)
                player.play(tts_b.audio, sample_rate=tts_b.sample_rate, blocking=True)
                time.sleep(0.3)
                print(f"   🔊 [2/2] Delivered to {name_a} in {info_a.name} [{lang_a.upper()}]...")
                tts_a = tts.synthesize(text=spoken_reply_for_a, language_code=lang_a, gender=args.gender_b)
                player.play(tts_a.audio, sample_rate=tts_a.sample_rate, blocking=True)
            else:
                tts_res = tts.synthesize(text=spoken_reply_for_a, language_code=lang_a, gender=args.gender_b)
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
                reply_in_b_raw = generate_call_partner_reply(
                    user_msg,
                    reply_lang=lang_b,
                    gemini_key=args.gemini_key,
                    user_name=name_a,
                    partner_name=name_b
                )
                if lang_b == "en":
                    reply_in_b = reply_in_b_raw
                else:
                    reply_in_b = translation.translate(reply_in_b_raw, source_lang="en", target_lang=lang_b).translated_text

                if lang_a == lang_b:
                    reply_for_a = reply_in_b
                else:
                    tr2 = translation.translate(reply_in_b, source_lang=lang_b, target_lang=lang_a)
                    reply_for_a = tr2.translated_text

                print(f"   👤 {name_b} ({info_b.name} [{lang_b.upper()}]): \"{reply_in_b}\"")
                if lang_a != lang_b:
                    print(f"   🌐 Translated to {info_a.name} [{lang_a.upper()}]: \"{reply_for_a}\"")

                if lang_b != lang_a:
                    print(f"   🔊 [1/2] Speaking {name_b}'s reply in {info_b.name} [{lang_b.upper()}]...")
                    tts_b = tts.synthesize(text=reply_in_b, language_code=lang_b, gender=args.gender_b)
                    player.play(tts_b.audio, sample_rate=tts_b.sample_rate, blocking=True)
                    time.sleep(0.3)
                    print(f"   🔊 [2/2] Delivered to {name_a} in {info_a.name} [{lang_a.upper()}]...")
                    tts_a = tts.synthesize(text=reply_for_a, language_code=lang_a, gender=args.gender_b)
                    player.play(tts_a.audio, sample_rate=tts_a.sample_rate, blocking=True)
                else:
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
