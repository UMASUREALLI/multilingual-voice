"""
Hugging Face Spaces Entry Point (Gradio SDK + FastAPI Backend).
Directly attaches FastAPI routes into Gradio's internal FastAPI app
so both the browser UI and Flutter mobile REST API endpoints run on port 7860 cleanly.
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

# Build clean status dashboard
with gr.Blocks(title="Multilingual Voice Call Translator") as demo:
    gr.Markdown("# 📞 Multilingual Voice AI Call Translator Backend")
    gr.Markdown(
        "✅ **The Real-Time Voice Translation API is Live and Operational!**\n\n"
        "This service powers real-time multilingual phone call translation for mobile apps (Flutter / React Native).\n\n"
        "- 📖 **Interactive API Documentation:** [Open Swagger UI (/docs)](/docs)\n"
        "- 💓 **Server Health Status:** [Check /health](/health)\n"
        "- 🌐 **Available Voices:** [Check /api/voices](/api/voices)\n"
    )

# Attach all FastAPI endpoints directly into Gradio's underlying ASGI app
for route in fastapi_app.routes:
    demo.app.routes.append(route)

if __name__ == "__main__":
    demo.launch()
