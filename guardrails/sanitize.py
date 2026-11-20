"""Neutralize untrusted text before it is shown to a language model."""

from __future__ import annotations

import re
import secrets
import unicodedata

MAX_FIELD_CHARS = 2_000
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_TAG = re.compile(r"</?\s*untrusted[^>]*>", re.IGNORECASE)


def sanitize_text(value: object, max_chars: int = MAX_FIELD_CHARS) -> str:
    """Strip control and invisible characters, defang delimiter tags, and truncate."""
    text = unicodedata.normalize("NFKC", str(value))
    text = _INVISIBLE.sub("", text)
    text = "".join(ch if ch in "\n\t" or unicodedata.category(ch)[0] != "C" else " " for ch in text)
    text = _TAG.sub("[tag removed]", text)
    return text if len(text) <= max_chars else text[:max_chars] + " [truncated]"


def wrap_untrusted(text: str) -> str:
    """Wrap text in a delimiter carrying a random nonce the log author cannot guess."""
    nonce = secrets.token_hex(8)
    return f'<untrusted id="{nonce}">\n{sanitize_text(text, 20_000)}\n</untrusted id="{nonce}">'
