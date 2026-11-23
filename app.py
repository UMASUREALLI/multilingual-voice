"""
Hugging Face Spaces Entry Point (Gradio SDK + FastAPI Backend).
Mounts the Multilingual Voice AI FastAPI endpoints seamlessly so the Flutter mobile team
can call all REST and WebSocket endpoints on the free Hugging Face tier (16 GB RAM).
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

# Create a clean status dashboard for visitors in the browser
with gr.Blocks(title="Multilingual Voice Call Translator") as demo:
    gr.Markdown("# 📞 Multilingual Voice AI Call Translator Backend")
    gr.Markdown(
        "✅ **The Real-Time Voice Translation API is Live and Operational!**\n\n"
        "This service powers real-time multilingual phone call translation for mobile apps (Flutter / React Native).\n\n"
        "- 📖 **Interactive API Documentation:** [Open Swagger UI (/docs)](/docs)\n"
        "- 💓 **Server Health Status:** [Check /health](/health)\n"
        "- 🌐 **Available Voices:** [Check /api/voices](/api/voices)\n"
    )

# Mount Gradio onto FastAPI so all /api/call/... endpoints and WebSockets remain active
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

# Also expose demo as a top-level symbol for Hugging Face Spaces Gradio runner
demo.app = app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
