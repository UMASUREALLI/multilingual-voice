"""
Named Entity and Technical Term Protection Engine.
Masks personal names, phone numbers, email addresses, technical acronyms,
and quantities before translation to prevent corrupted or hallucinated translations.
"""

import re
from typing import Tuple, Dict, List, Optional


class EntityPreserver:
    """
    Guards named entities and technical tokens from mistranslation.
    Replaces entities with unambiguous placeholders before MT, and restores them afterwards.
    """

    # Common technical terms to protect
    TECHNICAL_TERMS = {
        "AI", "ML", "API", "CPU", "GPU", "NPU", "VAD", "ASR", "TTS", "NMT",
        "FastAPI", "Python", "PyTorch", "Docker", "Node.js", "WebRTC", "Whisper",
        "React", "SQL", "HTML", "URL", "JSON", "REST", "SDK"
    }

    # Regex patterns for high-risk factual entities
    PATTERNS = [
        # Email addresses
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # URLs
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+\S*|www\.\S+",
        # Phone numbers (various international and Indian formats)
        r"(?:\+91[-.\s]?)?[6-9]\d{9}\b",
        r"\b\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
        # Currencies and quantities (e.g. $50, ₹1000, 10.5kg, 50%)
        r"(?:[\$₹€£]\s?\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:kg|km|gb|mb|am|pm|%)\b)",
        # Pure numeric strings of 3+ digits (account numbers, pin codes, OTPs)
        r"\b\d{3,}\b",
    ]

    def __init__(self, registered_names: Optional[List[str]] = None):
        self.registered_names = set(registered_names or [])

    def register_name(self, name: str) -> None:
        """Register a user name to preserve (e.g., caller or receiver name)."""
        clean = name.strip()
        if clean:
            self.registered_names.add(clean)

    def mask(
        self,
        text: str,
        additional_names: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Detects entities in text and replaces them with <<ENTITY_i>> placeholders.
        Returns: (masked_text, placeholder_map)
        """
        if not text:
            return text, {}

        placeholder_map: Dict[str, str] = {}
        masked_text = text
        entity_idx = 0

        # 1. Mask registered user names first
        all_names = self.registered_names.union(set(additional_names or []))
        for name in sorted(all_names, key=len, reverse=True):
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            matches = list(pattern.finditer(masked_text))
            for m in reversed(matches):
                orig_val = m.group(0)
                tag = f"<<ENTITY_{entity_idx}>>"
                placeholder_map[tag] = orig_val
                start, end = m.span()
                masked_text = masked_text[:start] + tag + masked_text[end:]
                entity_idx += 1

        # 2. Mask technical terms
        for term in sorted(self.TECHNICAL_TERMS, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            matches = list(pattern.finditer(masked_text))
            for m in reversed(matches):
                orig_val = m.group(0)
                tag = f"<<ENTITY_{entity_idx}>>"
                placeholder_map[tag] = orig_val
                start, end = m.span()
                masked_text = masked_text[:start] + tag + masked_text[end:]
                entity_idx += 1

        # 3. Mask regex patterns (phone numbers, emails, numbers, URLs)
        for pat in self.PATTERNS:
            regex = re.compile(pat, re.IGNORECASE)
            matches = list(regex.finditer(masked_text))
            for m in reversed(matches):
                orig_val = m.group(0)
                # Don't re-mask existing tags
                if orig_val.startswith("<<ENTITY_") and orig_val.endswith(">>"):
                    continue
                tag = f"<<ENTITY_{entity_idx}>>"
                placeholder_map[tag] = orig_val
                start, end = m.span()
                masked_text = masked_text[:start] + tag + masked_text[end:]
                entity_idx += 1

        return masked_text, placeholder_map

    def unmask(self, translated_text: str, placeholder_map: Dict[str, str]) -> str:
        """Restores original entities into the translated text."""
        if not translated_text:
            return translated_text

        import html
        # Unescape HTML entities that translation endpoints might introduce (e.g. &lt; -> <)
        result = html.unescape(translated_text)

        if not placeholder_map:
            return result

        for tag, orig_val in placeholder_map.items():
            bare_tag = tag.replace("<<", "").replace(">>", "")

            # Match exact tag <<ENTITY_0>>, or variants like <ENTITY_0>, (ENTITY_0), &lt;ENTITY_0&gt;, or bare ENTITY_0
            pattern = re.compile(
                rf"<<\s*{bare_tag}\s*>>|<\s*{bare_tag}\s*>|&lt;\s*{bare_tag}\s*&gt;|\(\s*{bare_tag}\s*\)|\b{bare_tag}\b",
                re.IGNORECASE
            )
            result = pattern.sub(orig_val, result)

        # Clean any leftover angle brackets around restored entities
        result = re.sub(r"<\s*([^<>]+?)\s*>", r"\1", result)
        result = re.sub(r"&lt;\s*([^&]+?)\s*&gt;", r"\1", result)

        return result
