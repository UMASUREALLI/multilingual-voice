"""
Full-Duplex Bidirectional Two-Speaker Call Simulator with Voice Cloning.
Simulates a real-time call between two speakers of different native languages:
Example: Speaker A speaks Telugu <-> Speaker B speaks Tamil.

Turn 1: Speaker A speaks Telugu
  -> System auto-identifies Telugu
  -> Translates Telugu -> Tamil for Speaker B
  -> Synthesizes Tamil speech in Speaker A's CLONED voice
  -> Speaker B hears Tamil spoken in Speaker A's voice!

Turn 2: Speaker B speaks Tamil
  -> System auto-identifies Tamil
  -> Translates Tamil -> Telugu for Speaker A
  -> Synthesizes Telugu speech in Speaker B's CLONED voice
  -> Speaker A hears Telugu spoken in Speaker B's voice!

Zero disk writes. Real-time VAD, LID, ASR, NMT, TTS, and acoustic feedback protection.
"""

import sys
import time
import argparse
import numpy as np

from app.audio.streamer import LiveAudioStreamer
from app.audio.vad import SpeechSegment
from app.audio.player import AudioPlaybackEngine, link_playback_to_streamer
from app.services.lid_service import LanguageDetectionEngine
from app.services.asr_service import StreamingAsrService
from app.services.translation_service import TranslationService
from app.services.tts_service import StreamingTtsService
from app.services.voice_cloner import VoiceCloningService
from app.models.language import SUPPORTED_LANGUAGES, LanguageInfo, normalize_language_code


def print_banner(a_name, a_lang, b_name, b_lang):
    print("\n" + "=" * 76)
    print("  BIDIRECTIONAL MULTILINGUAL VOICE CALL (SPEAKER A <-> SPEAKER B)")
    print(f"  Party A: {a_name} ({a_lang.upper()})  <== FULL-DUPLEX ==>  Party B: {b_name} ({b_lang.upper()})")
    print("=" * 76)
    print("  • When Speaker A speaks in their language:")
    print(f"    -> Translated to {b_lang.upper()} using Speaker A's CLONED voice for Speaker B.")
    print("  • When Speaker B speaks in their language:")
    print(f"    -> Translated to {a_lang.upper()} using Speaker B's CLONED voice for Speaker A.")
    print("=" * 76 + "\n")


# Canned realistic responses for auto-reply simulation when testing with a single user
AUTO_REPLIES = {
    "ta": [
        "வணக்கம்! நான் நன்றாக இருக்கிறேன், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "திட்டப்பணி மிகவும் சிறப்பாக முன்னேறி வருகிறது.",
        "நிச்சயமாக, நாம் நாளை காலை பேசலாம்."
    ],
    "te": [
        "నమస్కారం! నేను బాగున్నాను, మీరు ఎలా ఉన్నారు?",
        "ప్రాజెక్ట్ పని చాలా బాగా జరుగుతోంది.",
        "ఖచ్చితంగా, మనం రేపు ఉదయం మాట్లాడదాం."
    ],
    "hi": [
        "नमस्ते! मैं बहुत अच्छा हूँ, आप कैसे हैं?",
        "प्रोजेक्ट का काम बहुत तेजी से चल रहा है.",
        "ज़रूर, हम कल सुबह इस पर चर्चा करेंगे."
    ],
    "en": [
        "Hello! I am doing great, how about you?",
        "The project is progressing very well.",
        "Sure, let us schedule the follow-up meeting for tomorrow."
    ]
}


def main():
    parser = argparse.ArgumentParser(description="Bidirectional Two-Speaker Call Simulator")
    parser.add_argument("--party-a", type=str, default="te", help="Speaker A language (default: te / Telugu)")
    parser.add_argument("--party-b", type=str, default="ta", help="Speaker B language (default: ta / Tamil)")
    parser.add_argument("--gender-a", type=str, default="male", choices=["male", "female"], help="Speaker A gender")
    parser.add_argument("--gender-b", type=str, default="female", choices=["female", "male"], help="Speaker B gender")
    parser.add_argument("--auto-reply", action="store_true", default=True, help="Automatically simulate recipient reply")
    parser.add_argument("--no-auto-reply", dest="auto_reply", action="store_false", help="Disable auto-reply (user speaks both languages)")
    parser.add_argument("--pitch-a", type=str, default="+2Hz", help="Speaker A pitch modulation (voice clone)")
    parser.add_argument("--pitch-b", type=str, default="-3Hz", help="Speaker B pitch modulation (voice clone)")
    args = parser.parse_args()

    lang_a = normalize_language_code(args.party_a, default="te")
    lang_b = normalize_language_code(args.party_b, default="ta")

    info_a = SUPPORTED_LANGUAGES.get(lang_a, LanguageInfo(lang_a, lang_a.title(), lang_a))
    info_b = SUPPORTED_LANGUAGES.get(lang_b, LanguageInfo(lang_b, lang_b.title(), lang_b))

    name_a = f"Speaker A ({info_a.name})"
    name_b = f"Speaker B ({info_b.name})"

    print_banner(name_a, lang_a, name_b, lang_b)

    # 1. Initialize Engines
    print("Loading AI pipeline (Faster-Whisper INT8 CPU, Neural NMT, Neural TTS)...")
    from faster_whisper import WhisperModel
    shared_whisper = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

    lid_engine = LanguageDetectionEngine(whisper_model=shared_whisper, model_size="base")
    asr_service = StreamingAsrService(model_size="base", shared_model=shared_whisper)
    translation = TranslationService()
    tts = StreamingTtsService()
    player = AudioPlaybackEngine()

    streamer = LiveAudioStreamer(
        vad_start_threshold=0.45,
        vad_end_threshold=0.30,
        hangover_ms=500,
        min_speech_ms=300
    )
    # Echo feedback prevention
    link_playback_to_streamer(player, streamer, duck_only=False)

    print("Pipeline ready! Feedback loop protection active.\n")

    turn_count = 0
    reply_idx = 0

    def process_turn(audio_samples: np.ndarray, sample_rate: int = 16000):
        nonlocal turn_count, reply_idx
        turn_count += 1

        t0 = time.perf_counter()

        # Step 1: Detect spoken language
        t_lid = time.perf_counter()
        lid_res = lid_engine.identify_spoken(audio_samples, sample_rate=sample_rate)
        detected_lang = lid_res.language_code
        lid_ms = (time.perf_counter() - t_lid) * 1000.0

        # Step 2: Determine who spoke (Speaker A or Speaker B)
        # If detected language matches or clusters with Party A -> Speaker A spoke
        # If detected language matches or clusters with Party B -> Speaker B spoke
        if detected_lang == lang_b:
            active_speaker = name_b
            active_lang = lang_b
            active_gender = args.gender_b
            active_pitch = args.pitch_b

            recipient_speaker = name_a
            recipient_lang = lang_a
        else:
            active_speaker = name_a
            active_lang = lang_a
            active_gender = args.gender_a
            active_pitch = args.pitch_a

            recipient_speaker = name_b
            recipient_lang = lang_b

        # Step 3: ASR Speech-to-Text
        t_asr = time.perf_counter()
        asr_res = asr_service.transcribe(audio_or_path=audio_samples, language=detected_lang)
        asr_ms = (time.perf_counter() - t_asr) * 1000.0

        spoken_text = asr_res.text.strip()
        if not spoken_text:
            return

        # Step 4: Translation into recipient's language
        t_trans = time.perf_counter()
        trans_res = translation.translate(
            text=spoken_text,
            source_lang=detected_lang,
            target_lang=recipient_lang,
            speaker_name=active_speaker,
            receiver_name=recipient_speaker
        )
        trans_ms = (time.perf_counter() - t_trans) * 1000.0

        # Step 5: Neural TTS in recipient's language using ACTIVE SPEAKER'S CLONED VOICE
        t_tts = time.perf_counter()
        tts_res = tts.synthesize(
            text=trans_res.translated_text,
            language_code=recipient_lang,
            gender=active_gender,
            pitch=active_pitch
        )
        tts_ms = (time.perf_counter() - t_tts) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        # Display Live Call Log
        print("\n" + "=" * 76)
        print(f"📞 [CALL TURN #{turn_count}]")
        print(f"   👤 Spoken by     : {active_speaker} ({lid_res.language_name} [{detected_code_fmt(detected_lang)}])")
        print(f"   🗣️  Original Text : \"{spoken_text}\"")
        print(f"   ------------------------------------------------------------------------")
        print(f"   🎯 Delivered to  : {recipient_speaker} ({recipient_lang.upper()})")
        print(f"   🌐 Translation   : \"{trans_res.translated_text}\"")
        print(f"   🎙️ Cloned Voice  : {active_speaker}'s Voice ({tts_res.voice_id}, Pitch: {active_pitch})")
        print(f"   ⏱️ Latency       : {total_ms:.0f}ms (LID: {lid_ms:.0f}ms, ASR: {asr_ms:.0f}ms, NMT: {trans_ms:.0f}ms, TTS: {tts_ms:.0f}ms)")
        print(f"   🔊 Playing translation aloud for {recipient_speaker}...")
        print("=" * 76)

        # Play audio out loud for the recipient
        player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)

        # Step 6: If auto-reply is enabled and Speaker A spoke, simulate Speaker B's response!
        if args.auto_reply and active_speaker == name_a:
            time.sleep(0.5)
            simulate_recipient_reply(
                speaker_name=name_b,
                speaker_lang=lang_b,
                speaker_gender=args.gender_b,
                speaker_pitch=args.pitch_b,
                target_name=name_a,
                target_lang=lang_a
            )

    def simulate_recipient_reply(speaker_name, speaker_lang, speaker_gender, speaker_pitch, target_name, target_lang):
        nonlocal turn_count, reply_idx
        turn_count += 1

        replies = AUTO_REPLIES.get(speaker_lang, AUTO_REPLIES["ta"])
        reply_text = replies[reply_idx % len(replies)]
        reply_idx += 1

        t0 = time.perf_counter()
        # Translate B's reply to A's language
        t_trans = time.perf_counter()
        trans_res = translation.translate(
            text=reply_text,
            source_lang=speaker_lang,
            target_lang=target_lang,
            speaker_name=speaker_name,
            receiver_name=target_name
        )
        trans_ms = (time.perf_counter() - t_trans) * 1000.0

        # Synthesize in A's language using SPEAKER B'S CLONED VOICE
        t_tts = time.perf_counter()
        tts_res = tts.synthesize(
            text=trans_res.translated_text,
            language_code=target_lang,
            gender=speaker_gender,
            pitch=speaker_pitch
        )
        tts_ms = (time.perf_counter() - t_tts) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        print("\n" + "=" * 76)
        print(f"📞 [CALL TURN #{turn_count} - RESPONSE]")
        print(f"   👤 Spoken by     : {speaker_name} ({speaker_lang.upper()})")
        print(f"   🗣️  Original Text : \"{reply_text}\"")
        print(f"   ------------------------------------------------------------------------")
        print(f"   🎯 Delivered to  : {target_name} ({target_lang.upper()})")
        print(f"   🌐 Translation   : \"{trans_res.translated_text}\"")
        print(f"   🎙️ Cloned Voice  : {speaker_name}'s Voice ({tts_res.voice_id}, Pitch: {speaker_pitch})")
        print(f"   ⏱️ Latency       : {total_ms:.0f}ms (NMT: {trans_ms:.0f}ms, TTS: {tts_ms:.0f}ms)")
        print(f"   🔊 Playing translation aloud for {target_name}...")
        print("=" * 76)

        player.play(tts_res.audio, sample_rate=tts_res.sample_rate, blocking=True)
        print(f"\n[Microphone Active] Listening for {name_a} to speak again...")

    def detected_code_fmt(code):
        return code.upper()

    streamer.register_on_utterance_complete(lambda seg: process_turn(seg.audio, seg.sample_rate))
    streamer.start()

    print("🟢 CALL IN PROGRESS!")
    print(f"👉 Speak into your microphone in {info_a.name} ({lang_a}) as {name_a}.")
    print(f"   {name_b} will hear your speech in {info_b.name} ({lang_b}) in YOUR cloned voice.")
    print("   Press Ctrl+C to terminate the call.\n")

    try:
        spinner = ["|", "/", "-", "\\"]
        idx = 0
        while True:
            rms = streamer.current_rms_dbfs
            prob = streamer.current_speech_prob
            is_speaking = prob >= 0.45
            tag = "🗣️ SPEAKING" if is_speaking else "👂 LISTENING"
            bar_len = int(max(0, min(14, (rms + 60) / 4)))
            vu_bar = "█" * bar_len + "░" * (14 - bar_len)
            muted = " [MIC DUCKED]" if player.is_playing else ""

            sys.stdout.write(f"\r{spinner[idx % 4]} [{tag}] [{vu_bar}] {rms:5.1f} dBFS | VAD: {prob:4.2f}{muted}   ")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\n\nTerminating call session...")
    finally:
        streamer.stop()
        player.stop()

    print(f"Call session ended. Total turns: {turn_count}\n")


if __name__ == "__main__":
    main()
