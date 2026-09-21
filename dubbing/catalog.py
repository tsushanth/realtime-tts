"""Enumerate target languages actually available from voices/catalog.json.

Dubbing must never claim support for a language we can't actually synthesize. This is the
single source of truth other modules import from - no hardcoded language lists elsewhere.
"""
import json
import os

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "voices", "catalog.json")


def load_catalog(path: str = CATALOG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def voices_by_language(path: str = CATALOG_PATH) -> dict[str, list[dict]]:
    """{"de_DE": [voice, voice, ...], ...} - every voice actually present in the catalog."""
    cat = load_catalog(path)
    out: dict[str, list[dict]] = {}
    for v in cat["voices"]:
        out.setdefault(v["language"], []).append(v)
    return out


def supported_languages(path: str = CATALOG_PATH) -> list[str]:
    """BCP-47-ish language tags (e.g. 'de_DE') we can actually dub into right now."""
    return sorted(voices_by_language(path).keys())


def best_voice_for(language: str, path: str = CATALOG_PATH) -> dict:
    """Pick a default voice for a language: prefer tier A, then lowest cpu_rtf (fastest)."""
    by_lang = voices_by_language(path)
    if language not in by_lang:
        raise ValueError(
            f"target language {language!r} is not in voices/catalog.json. "
            f"Supported: {sorted(by_lang.keys())}"
        )
    candidates = by_lang[language]
    tier_a = [v for v in candidates if v.get("tier") == "A"]
    pool = tier_a or candidates
    return min(pool, key=lambda v: v.get("cpu_rtf", 999))


def validate_target_language(language: str, path: str = CATALOG_PATH) -> None:
    supported = supported_languages(path)
    if language not in supported:
        raise ValueError(
            f"Unsupported target language {language!r}. Only languages present in "
            f"voices/catalog.json are supported for dubbing: {supported}"
        )


if __name__ == "__main__":
    for lang in supported_languages():
        v = best_voice_for(lang)
        print(f"{lang:8s} default voice: {v['id']} (tier {v['tier']}, cpu_rtf {v['cpu_rtf']})")
