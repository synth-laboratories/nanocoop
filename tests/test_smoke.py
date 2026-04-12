from nanocoop.baselines import offline_sft, prompt_opt, rlvr
from nanocoop.io import load_yaml


def test_offline_smoke_runs():
    result = offline_sft.run(load_yaml("configs/offline_smoke.yaml"))
    assert result["metrics"]["num_eval_episodes"] > 0
    assert "primary_score" in result["metrics"]


def test_rlvr_smoke_runs():
    result = rlvr.run(load_yaml("configs/rlvr_smoke.yaml"))
    assert result["metrics"]["num_eval_episodes"] > 0
    assert "primary_score" in result["metrics"]


def test_prompt_opt_smoke_runs():
    result = prompt_opt.run(load_yaml("configs/prompt_opt_smoke.yaml"))
    assert result["metrics"]["num_eval_episodes"] > 0
    assert "primary_score" in result["metrics"]
