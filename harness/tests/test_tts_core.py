import json
import os
import tempfile

from harness.recipe import Candidate
from harness.recipes.tts_core import KNOWN_SIZE_VARIANTS, TTSCoreRecipe


def test_discover_candidates_returns_known_size_variants_not_yet_tried(tmp_path, monkeypatch):
    tried_path = tmp_path / "tried.json"
    tried_path.write_text(json.dumps({}))
    monkeypatch.setattr("harness.recipes.tts_core.TRIED_PATH", str(tried_path))

    recipe = TTSCoreRecipe()
    candidates = recipe.discover_candidates()

    assert len(candidates) == len(KNOWN_SIZE_VARIANTS)
    ids = {c.id for c in candidates}
    assert ids == {v["size_label"] for v in KNOWN_SIZE_VARIANTS}
    for c in candidates:
        assert set(c.train_config.keys()) == {"warmstart_url", "max_steps", "size_label"}


def test_discover_candidates_excludes_already_tried(tmp_path, monkeypatch):
    already_tried_id = KNOWN_SIZE_VARIANTS[0]["size_label"]
    tried_path = tmp_path / "tried.json"
    tried_path.write_text(json.dumps({already_tried_id: {"date": "2026-01-01", "report_path": "x.md"}}))
    monkeypatch.setattr("harness.recipes.tts_core.TRIED_PATH", str(tried_path))

    recipe = TTSCoreRecipe()
    candidates = recipe.discover_candidates()

    ids = {c.id for c in candidates}
    assert already_tried_id not in ids
    assert len(candidates) == len(KNOWN_SIZE_VARIANTS) - 1


def test_estimate_cost_scales_with_max_steps():
    recipe = TTSCoreRecipe()
    small = Candidate(id="small", description="", train_config={"warmstart_url": "x", "max_steps": 5000, "size_label": "small"})
    large = Candidate(id="large", description="", train_config={"warmstart_url": "x", "max_steps": 20000, "size_label": "large"})

    small_cost = recipe.estimate_cost_usd(small)
    large_cost = recipe.estimate_cost_usd(large)

    assert large_cost > small_cost
    # Basis: piper_full_finetune.py's own measured rate, ~1.2 it/s at batch_size=8
    # on a T4 ($0.59/hr per Modal's published T4 rate at time of writing -
    # verify against `modal pricing` or the Modal dashboard before trusting this
    # exact number for a real budget decision), giving 20000 steps ~= 4.6h ~= $2.50-3,
    # matching the docstring in training-data/piper_full_finetune.py.
    assert 2.0 <= large_cost <= 4.0


def test_discover_candidates_handles_corrupted_tried_json(tmp_path, monkeypatch):
    """Verify that corrupted/truncated JSON in tried-file is handled gracefully
    without crashing, falling back to empty dict (all variants untried)."""
    tried_path = tmp_path / "tried.json"
    # Write malformed JSON (incomplete, truncated mid-flight)
    tried_path.write_text('{"low": {"date": "2026-01-01"')
    monkeypatch.setattr("harness.recipes.tts_core.TRIED_PATH", str(tried_path))

    recipe = TTSCoreRecipe()
    # Should not raise; instead, should degrade gracefully to treating file as empty
    candidates = recipe.discover_candidates()

    # All variants should be returned since corrupted file is treated as "none tried yet"
    assert len(candidates) == len(KNOWN_SIZE_VARIANTS)
    ids = {c.id for c in candidates}
    assert ids == {v["size_label"] for v in KNOWN_SIZE_VARIANTS}
