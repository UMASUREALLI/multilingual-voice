"""
Dynamic Conversational AI Dialogue Service for Multilingual Call Partner.
Generates 100% open-ended, contextual, dynamic neural responses for any spoken input.
No static answers. No hardcoded if-else statements.

Two engines available:
1. Local Neural Small Language Model (SLM): Qwen2.5-0.5B-Instruct (runs 100% offline on CPU)
2. Cloud Generative LLM: Google Gemini 2.0 Flash / 1.5 Flash (via free API key)
"""

import os
import re
import json
import logging
import urllib.request
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("call_partner.dialogue")


class DynamicDialogueService:
    """
    Real-time dynamic conversational AI generator for call partner (Sarah).
    Generates genuine, non-static, contextual spoken dialogue token-by-token.
    """

    _local_tokenizer = None
    _local_model = None

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        local_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        partner_name: str = "Sarah",
        user_name: str = "Uma"
    ):
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.local_model_name = local_model_name
        self.partner_name = partner_name
        self.user_name = user_name

        # Conversation history for multi-turn conversational context
        self.history: List[Dict[str, str]] = []

    @classmethod
    def load_local_engine(cls, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """Pre-loads local LLM weights into memory once."""
        if cls._local_model is None:
            logger.info(f"Loading local neural dialogue model '{model_name}' on CPU...")
            cls._local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls._local_model = AutoModelForCausalLM.from_pretrained(model_name)
            logger.info("Local dialogue model ready.")

    def _query_gemini(self, user_text: str, reply_lang: str = "en") -> Optional[str]:
        """Queries Google Gemini 2.0 / 1.5 Flash for lightning-fast dynamic responses."""
        if not self.gemini_api_key:
            return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            prompt = (
                f"You are {self.partner_name}, a friendly, intelligent AI voice call partner having a live call with {self.user_name}. "
                f"Respond directly, contextually, and conversationally to what {self.user_name} says in 1 or 2 natural spoken sentences. "
                f"Language: {reply_lang}. "
                f"Do not use markdown formatting, asterisks, emojis, bullet points, or stage directions. Spoken voice text only.\n\n"
                f"{self.user_name}: {user_text}\n"
                f"{self.partner_name}:"
            )
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 60, "temperature": 0.7}
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                clean = re.sub(r"[\*\_\"\#\n]+", " ", text).strip()
                if clean:
                    return clean
        except Exception as e:
            logger.warning(f"Gemini API query failed, falling back to local LLM: {e}")
        return None

    def _query_local_llm(self, user_text: str, reply_lang: str = "en") -> str:
        """Generates dynamic neural response using Qwen2.5-0.5B-Instruct on CPU."""
        if self._local_model is None:
            self.load_local_engine(self.local_model_name)

        tokenizer = self._local_tokenizer
        model = self._local_model

        system_instruction = (
            f"You are {self.partner_name}, a warm and helpful AI voice call partner talking to {self.user_name}. "
            f"Always reply in 1-2 short, natural spoken sentences. "
            f"Do not use emojis, markdown, or lists. Reply in spoken English."
        )

        messages = [
            {"role": "system", "content": system_instruction},
        ]
        # Include last 2 turns of context for natural dialogue flow
        for item in self.history[-4:]:
            messages.append(item)

        messages.append({"role": "user", "content": user_text})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=45,
                do_sample=True,
                temperature=0.75,
                top_p=0.90,
                repetition_penalty=1.15
            )

        new_tokens = generated_ids[:, model_inputs.input_ids.shape[1]:]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        # Clean formatting
        clean_response = re.sub(r"[\*\_\"\#\n]+", " ", response).strip()
        # Keep to first 2 sentences if model generates more
        sentences = re.split(r"(?<=[.!?]) +", clean_response)
        short_response = " ".join(sentences[:2]).strip()

        return short_response if short_response else clean_response

    def generate_reply(self, user_text: str, reply_lang: str = "en") -> str:
        """
        Dynamically generates a contextual conversational response.
        Prioritizes Gemini if key is provided; otherwise generates via local neural SLM on CPU.
        """
        # 1. Try Gemini cloud LLM first if key available
        reply = self._query_gemini(user_text, reply_lang=reply_lang)

        # 2. If no cloud key, run local Qwen neural model on CPU
        if not reply:
            reply = self._query_local_llm(user_text, reply_lang=reply_lang)

        # Record history for multi-turn awareness
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

        return reply
