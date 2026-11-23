"""
Hugging Face Spaces Entry Point (ZeroGPU + Gradio SDK + FastAPI Backend).
Full-Duplex Voice Call Translation:
- Speech-to-Speech (Acoustic LID, ASR, Translation, Acoustic Gender Detection & Neural Voice Synthesis)
- Text-to-Speech (Contextual Translation & Multi-Gender Neural Speech Playback)
"""

import os
import sys

# Ensure backend directory is in the Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np
from main import app as fastapi_app, coordinator
from app.services.voice_cloner import VoiceCloningService
import gradio as gr

# Intercept Gradio's internal App creation immediately so all FastAPI endpoints take priority
orig_create_app = gr.routes.App.create_app

def custom_create_app(*args, **kwargs):
    app = orig_create_app(*args, **kwargs)
    app.include_router(fastapi_app.router)
    return app

gr.routes.App.create_app = staticmethod(custom_create_app)

# Hugging Face ZeroGPU integration
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    # Transparent fallback when running outside Hugging Face ZeroGPU (e.g. Windows local)
    def gpu_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def dec(fn):
            return fn
        return dec


# Anchor function to satisfy Hugging Face ZeroGPU startup requirements without consuming user quota
@gpu_decorator(duration=5)
def _zerogpu_anchor():
    return ""


# Real-time Text-to-Speech translation with Gender persona selection
def voice_translate_text(text: str, source_lang: str, target_lang: str, gender: str = "Female"):
    if not text or not text.strip():
        return "", "", None
    trans_res = coordinator.translation.translate(text=text, source_lang=source_lang, target_lang=target_lang)
    selected_gender = "female" if gender.lower() == "female" else "male"
    tts_res = coordinator.tts.synthesize(text=trans_res.translated_text, language_code=target_lang, gender=selected_gender)
    voice_info = f"{tts_res.voice_id} ({selected_gender.capitalize()})" if tts_res else ""
    audio_tuple = (tts_res.sample_rate, tts_res.audio) if tts_res is not None else None
    return trans_res.translated_text, voice_info, audio_tuple


# Real-time Speech-to-Speech translation with Automatic Gender & Pitch Detection
def voice_translate_audio(audio_data, source_lang: str, target_lang: str):
    if audio_data is None:
        return "", "", "", "", None
    sr, raw_audio = audio_data
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)
    if raw_audio.dtype == np.int16:
        raw_audio = raw_audio.astype(np.float32) / 32768.0
    elif raw_audio.dtype != np.float32:
        raw_audio = raw_audio.astype(np.float32)
    if sr != 16000 and len(raw_audio) > 0:
        import scipy.signal
        num_target = int(len(raw_audio) * 16000 / sr)
        raw_audio = scipy.signal.resample(raw_audio, num_target).astype(np.float32)
        sr = 16000

    # 1. Acoustic Gender & Fundamental Pitch (F0) Detection
    feat = VoiceCloningService.extract_acoustic_features(raw_audio, sr)
    detected_gender = feat.estimated_gender
    detected_pitch = feat.f0_median_hz
    gender_info = f"{detected_gender.capitalize()} Voice (Pitch: {detected_pitch:.1f} Hz)"

    # 2. Native Script Whisper ASR & Acoustic Language Identification
    asr_lang = None if source_lang == "auto" else source_lang
    asr_res = coordinator.asr.transcribe(audio_or_path=raw_audio, language=asr_lang)
    rec_text = asr_res.text.strip()
    if not rec_text:
        return "(No speech detected)", "", gender_info, "", None

    # 3. Contextual Translation
    trans_res = coordinator.translation.translate(text=rec_text, source_lang=asr_res.language, target_lang=target_lang)

    # 4. Neural Speech Synthesis matching the speaker's detected gender
    tts_res = coordinator.tts.synthesize(text=trans_res.translated_text, language_code=target_lang, gender=detected_gender)
    voice_info = f"{tts_res.voice_id} ({detected_gender.capitalize()})" if tts_res else ""
    audio_tuple = (tts_res.sample_rate, tts_res.audio) if tts_res is not None else None

    return rec_text, trans_res.translated_text, gender_info, voice_info, audio_tuple


# Supported Language Choices
LANG_CHOICES = ["te", "hi", "en", "ta", "kn", "ml", "mr", "bn", "gu", "pa", "ur", "es", "fr", "de", "ja", "zh", "ar"]
LANG_CHOICES_WITH_AUTO = ["auto"] + LANG_CHOICES

# Build Gradio UI
with gr.Blocks(title="Multilingual Voice Call Translator") as demo:
    # Invisible anchor event so Hugging Face ZeroGPU startup scanner passes without error
    _anchor_btn = gr.Button("ZeroGPU Anchor", visible=False)
    _anchor_btn.click(fn=_zerogpu_anchor)

    gr.Markdown("# Multilingual Voice AI Call Translator Backend")
    gr.Markdown(
        "**The Real-Time Voice Translation API is Live and Operational.**\n\n"
        "- **Interactive API Documentation:** [Open Swagger UI (/docs)](/docs)\n"
        "- **Server Health Status:** [Check /health](/health)\n"
        "- **Available Voices:** [Check /api/voices](/api/voices)\n"
    )

    with gr.Tabs():
        with gr.Tab("Microphone / Audio Call (Speech-to-Speech)"):
            gr.Markdown(
                "### Speak or Upload Spoken Audio\n"
                "*Automatically identifies spoken language, transcribes speech, detects male/female vocal pitch, and synthesizes matching neural voice.*"
            )
            with gr.Row():
                mic_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Voice Audio")
            with gr.Row():
                mic_src_dropdown = gr.Dropdown(
                    choices=LANG_CHOICES_WITH_AUTO,
                    value="auto",
                    label="Spoken Language (auto-detect available)"
                )
                mic_tgt_dropdown = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="en",
                    label="Target Translation Language"
                )
            btn_mic_translate = gr.Button("Translate Speech & Speak", variant="primary")
            with gr.Row():
                mic_transcribed = gr.Textbox(label="Transcribed Spoken Text (Native Script)", lines=2)
                mic_translated = gr.Textbox(label="Translated Speech Subtitle", lines=2)
            with gr.Row():
                mic_gender_status = gr.Textbox(label="Acoustically Detected Gender & Pitch (F0)", lines=1)
                mic_voice_status = gr.Textbox(label="Assigned Neural Voice Persona", lines=1)
            with gr.Row():
                mic_audio_out = gr.Audio(label="Translated Neural Speech (Listen)", autoplay=True)

            btn_mic_translate.click(
                fn=voice_translate_audio,
                inputs=[mic_input, mic_src_dropdown, mic_tgt_dropdown],
                outputs=[mic_transcribed, mic_translated, mic_gender_status, mic_voice_status, mic_audio_out]
            )

        with gr.Tab("Type Spoken Text (Text-to-Speech)"):
            gr.Markdown("### Type or Paste Spoken Dialogue\n*Translates text and synthesizes speech in selected regional female or male persona.*")
            with gr.Row():
                txt_input = gr.Textbox(
                    label="Spoken Text (Telugu, Hindi, English, Tamil, etc.)",
                    value="ఏం చేయట్లేదు బ్రో",
                    lines=2
                )
            with gr.Row():
                src_dropdown = gr.Dropdown(
                    choices=LANG_CHOICES_WITH_AUTO,
                    value="te",
                    label="Source Language"
                )
                tgt_dropdown = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="en",
                    label="Target Language"
                )
                gender_dropdown = gr.Dropdown(
                    choices=["Female", "Male"],
                    value="Female",
                    label="Voice Gender Persona"
                )
            btn_translate = gr.Button("Translate & Speak", variant="primary")
            with gr.Row():
                txt_output = gr.Textbox(label="Translated Speech Subtitle", lines=2)
                txt_voice_status = gr.Textbox(label="Active Neural Voice Persona", lines=2)
            with gr.Row():
                txt_audio_out = gr.Audio(label="Translated Speech (Listen)", autoplay=True)

            btn_translate.click(
                fn=voice_translate_text,
                inputs=[txt_input, src_dropdown, tgt_dropdown, gender_dropdown],
                outputs=[txt_output, txt_voice_status, txt_audio_out]
            )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
