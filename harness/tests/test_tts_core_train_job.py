from harness.recipes.tts_core_train_job import build_piper_train_args


def test_build_piper_train_args_uses_candidate_config():
    args = build_piper_train_args(
        warmstart_url="https://example.com/checkpoint.ckpt",
        max_steps=8000,
        size_label="low",
        warmstart_local_path="/ckpt/warmstart.ckpt",
    )

    assert "--model.warmstart_ckpt" in args
    idx = args.index("--model.warmstart_ckpt")
    assert args[idx + 1] == "/ckpt/warmstart.ckpt"

    assert "--trainer.max_steps" in args
    idx = args.index("--trainer.max_steps")
    assert args[idx + 1] == "8000"

    assert "--trainer.default_root_dir" in args
    idx = args.index("--trainer.default_root_dir")
    assert "low" in args[idx + 1]  # size_label keeps different candidates' checkpoints from colliding


def test_build_piper_train_args_different_sizes_get_different_output_dirs():
    args_low = build_piper_train_args(warmstart_url="x", max_steps=8000, size_label="low", warmstart_local_path="/ckpt/x.ckpt")
    args_high = build_piper_train_args(warmstart_url="x", max_steps=20000, size_label="high", warmstart_local_path="/ckpt/x.ckpt")

    idx_low = args_low.index("--trainer.default_root_dir")
    idx_high = args_high.index("--trainer.default_root_dir")
    assert args_low[idx_low + 1] != args_high[idx_high + 1]


from unittest.mock import MagicMock, patch

from harness.recipe import Candidate
from harness.recipes.tts_core import TTSCoreRecipe


def test_train_spawns_and_waits_on_modal_call():
    candidate = Candidate(
        id="low",
        description="test",
        train_config={"warmstart_url": "https://example.com/x.ckpt", "max_steps": 100, "size_label": "low"},
    )

    fake_call = MagicMock()
    fake_call.get.return_value = "/checkpoints/harness_tts_core_low"

    with patch("harness.recipes.tts_core_train_job.run_piper_finetune") as fake_fn:
        fake_fn.spawn.return_value = fake_call
        recipe = TTSCoreRecipe()
        result = recipe.train(candidate, workdir="/tmp/workdir")

    fake_fn.spawn.assert_called_once_with(warmstart_url="https://example.com/x.ckpt", max_steps=100, size_label="low")
    fake_call.get.assert_called_once()
    assert result.artifact_path == "/checkpoints/harness_tts_core_low"
    assert result.candidate is candidate
    assert result.actual_cost_usd >= 0.0
