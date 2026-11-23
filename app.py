"""
Hugging Face Spaces Entry Point (ZeroGPU + Gradio SDK + FastAPI Backend).
Decorates Gradio functions with @spaces.GPU to satisfy Hugging Face ZeroGPU requirements,
while directly serving all FastAPI REST and WebSocket call translation endpoints.
"""

import os
import sys

# Ensure backend directory is in the Python search path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app as fastapi_app
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
    def gpu_decorator(fn):
        return fn


# Top-level GPU-accelerated translation function for Gradio interface
@gpu_decorator
def translate_message(text: str, source_lang: str, target_lang: str) -> str:
    """ZeroGPU accelerated translation endpoint for browser testing."""
    if not text.strip():
        return ""
    from app.services.translation_service import TranslationService
    service = TranslationService()
    res = service.translate(text=text, source_lang=source_lang, target_lang=target_lang)
    return res.translated_text


# Build Gradio UI with bound ZeroGPU handler
with gr.Blocks(title="Multilingual Voice Call Translator") as demo:
    gr.Markdown("# Multilingual Voice AI Call Translator Backend")
    gr.Markdown(
        "**The Real-Time Voice Translation API is Live and Operational.**\n\n"
        "- **Interactive API Documentation:** [Open Swagger UI (/docs)](/docs)\n"
        "- **Server Health Status:** [Check /health](/health)\n"
        "- **Available Voices:** [Check /api/voices](/api/voices)\n"
    )

    with gr.Row():
        txt_input = gr.Textbox(
            label="Spoken Text (Telugu, Hindi, English, Tamil, etc.)",
            value="ఏం చేయట్లేదు బ్రో",
            lines=2
        )
        src_dropdown = gr.Dropdown(
            choices=["te", "hi", "en", "ta", "kn", "auto"],
            value="te",
            label="Source Language"
        )
        tgt_dropdown = gr.Dropdown(
            choices=["en", "te", "hi", "ta", "kn"],
            value="en",
            label="Target Language"
        )

    btn_translate = gr.Button("Translate Now", variant="primary")
    txt_output = gr.Textbox(label="Translated Speech Subtitle", lines=2)

    # Bind button directly to @spaces.GPU function so ZeroGPU startup check passes!
    btn_translate.click(
        fn=translate_message,
        inputs=[txt_input, src_dropdown, tgt_dropdown],
        outputs=txt_output
    )

if __name__ == "__main__":
    demo.launch()


