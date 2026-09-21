"""Translation step: source text -> target-language text, budgeted for spoken duration.

Dubbing lives or dies on the translated text being roughly as long to *speak* as the source
segment (retiming/stretch downstream can only close a bounded gap, see retime.py's +-15%
cap) - so the LLM prompt carries an explicit character/duration budget, not just "translate
this", and is asked to prefer a rephrasing that fits over a literal one that doesn't.

Interface is provider-agnostic (`Translator` ABC) so the LLM backend is swappable. Only one
concrete backend is wired up: OpenRouterTranslator, because OPENROUTER_API_KEY is the only LLM
credential actually present in this environment (checked directly, not assumed - see
dubbing/README.md). No other provider key (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.) was found
configured, so no other backend is wired. If OPENROUTER_API_KEY is absent at runtime,
StubTranslator raises clearly rather than silently passing text through or hardcoding a key.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Chars-per-second is a rough spoken-rate proxy shared with gateway/keys.js's STT_CHARS_PER_SECOND
# billing constant (3.0556 chars/s median English speech) - reused here only as a sanity anchor for
# the prompt budget, not for billing.
CHARS_PER_SECOND_ANCHOR = 3.0556


@dataclass
class TranslationResult:
    text: str
    source_lang: str
    target_lang: str
    source_chars: int
    target_chars: int
    budget_chars: int
    model: str


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str, duration_s: float | None = None) -> TranslationResult:
        ...


class StubTranslator(Translator):
    """Explicit non-functional stand-in. Raises instead of pretending to translate, so a missing
    API key fails loudly at the translation step rather than shipping source-language text
    mislabeled as dubbed audio."""

    def __init__(self, reason: str):
        self.reason = reason

    def translate(self, text, source_lang, target_lang, duration_s=None):
        raise RuntimeError(
            "Translation is stubbed: no LLM translation API is configured "
            f"({self.reason}). Provision an API key (e.g. OPENROUTER_API_KEY, or wire up "
            "another backend behind the Translator interface in dubbing/translate.py) before "
            "running the pipeline end to end. Refusing to silently pass text through untranslated."
        )


class OpenRouterTranslator(Translator):
    """Real backend, used because OPENROUTER_API_KEY is the only LLM credential present in this
    environment. Any OpenRouter chat model works; default is a small fast one since this is a
    short-segment translation task, not open-ended generation."""

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.model = model

    def translate(self, text: str, source_lang: str, target_lang: str, duration_s: float | None = None) -> TranslationResult:
        source_chars = len(text)
        # Budget: match source character count by default (same-order-of-magnitude speaking time
        # across languages is a weak proxy, but a cheap one), or duration_s * chars/s if we know the
        # clip's actual spoken duration from STT.
        budget_chars = int(round(duration_s * CHARS_PER_SECOND_ANCHOR)) if duration_s else source_chars
        # +-15% slack matches retime.py's bounded time-stretch cap: the LLM should aim inside a
        # window that a final ffmpeg atempo pass can still close without audible artifacts.
        lo, hi = int(budget_chars * 0.85), int(budget_chars * 1.15)

        prompt = (
            f"Translate the following {source_lang} text into {target_lang} for AUDIO DUBBING "
            f"(text-to-speech narration only - no video, no lip-sync).\n\n"
            f"Hard constraint: the translated text, when spoken aloud, must take roughly as long "
            f"as the source ({budget_chars} characters is the target length; acceptable range "
            f"{lo}-{hi} characters). If the source language is normally more/less verbose than the "
            f"target language for this content, REPHRASE to fit the budget rather than translating "
            f"literally - prefer a shorter idiomatic phrasing over a longer literal one, and vice "
            f"versa. Do not drop meaning to hit the budget; compress phrasing instead.\n\n"
            f"Return ONLY the translated text, no notes, no quotes, no language labels.\n\n"
            f"Source text:\n{text}"
        )

        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }).encode("utf-8")

        req = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # OpenRouter asks for these identifying headers; harmless, local dev values.
                "HTTP-Referer": "https://github.com/local/realtime-tts-dubbing-mvp",
                "X-Title": "dubbing-mvp",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"OpenRouter translation request failed: {e.code} {e.read().decode(errors='replace')}") from e

        translated = data["choices"][0]["message"]["content"].strip()
        return TranslationResult(
            text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            source_chars=source_chars,
            target_chars=len(translated),
            budget_chars=budget_chars,
            model=self.model,
        )


def get_default_translator() -> Translator:
    """Picks OpenRouter if configured, else a StubTranslator that fails loudly and says why."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterTranslator()
    return StubTranslator("OPENROUTER_API_KEY not set in environment")


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "Thanks for calling, how can I help you today?"
    src = sys.argv[2] if len(sys.argv) > 2 else "en"
    tgt = sys.argv[3] if len(sys.argv) > 3 else "de"
    t = get_default_translator()
    r = t.translate(text, src, tgt)
    print(json.dumps(r.__dict__, indent=2, ensure_ascii=False))
