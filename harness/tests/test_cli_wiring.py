"""Test CLI recipe registry wiring."""
from harness.recipes import get_recipe
import harness.recipes.tts_core  # noqa: F401 - import triggers registration


def test_tts_core_is_registered_by_default():
    recipe = get_recipe("tts-core")
    assert recipe.name == "tts-core"


def test_unknown_recipe_raises_with_helpful_message():
    import pytest
    with pytest.raises(ValueError, match="Unknown recipe"):
        get_recipe("does-not-exist")
