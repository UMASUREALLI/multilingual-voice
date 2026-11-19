import os
from faster_whisper import WhisperModel
from app.services.lid_service import LanguageDetectionEngine

print("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
engine = LanguageDetectionEngine(whisper_model=model)

files = [
    r"D:\uma surepalli\Offic_Projects\multilingual-voice\backend\microphone_test.wav",
    r"C:\Users\user\Documents\Sound recordings\Recording (autosaved).m4a",
    r"D:\uma surepalli\Offic_Projects\pan-india-voice-bridge\data\transcripts\demo_spoken_sample.wav",
]

for p in files:
    if os.path.exists(p):
        print(f"\n==========================================")
        print(f"Testing File: {os.path.basename(p)}")
        print(f"Path: {p}")
        try:
            segments, info = model.transcribe(p, language=None, beam_size=1)
            segs = list(segments)
            text = " ".join(s.text.strip() for s in segs)
            print(f"Acoustic Spoken LID : {info.language} ({info.language_probability:.1%})")
            print(f"Transcribed Text    : '{text}'")

            script_decision = engine.identify_script(text)
            if script_decision:
                print(f"Script LID Decision : {script_decision.language_name} ({script_decision.language_code}) at {script_decision.confidence:.1%}")
        except Exception as e:
            print(f"Error processing {p}: {e}")
    else:
        print(f"File not found: {p}")
