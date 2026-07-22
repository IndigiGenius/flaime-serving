"""Tests for the two extracted vendored stubs (26Q3-REPO-01).

These mirror the upstream behavior of PackingConfig
(flaime/infrastructure/datasets/dynamic_batching.py) and is_main_process
(flaime/utils/distributed.py) so drift from the recorded FLAIME snapshot
would surface here as well as in the tamper gate.
"""

from typing import Any

import pytest

from flaime_serving.vendored.distributed import get_rank_and_world_size, is_main_process
from flaime_serving.vendored.packing_config import PackingConfig


class TestPackingConfig:
    def test_defaults(self) -> None:
        config = PackingConfig()
        assert config.max_duration_seconds == 30.0
        assert config.max_samples_per_pack == 10
        assert config.sample_rate == 16000
        assert config.gap_seconds == 0.1
        assert config.enabled is True

    def test_derived_properties(self) -> None:
        config = PackingConfig(
            max_duration_seconds=2.0, sample_rate=16000, gap_seconds=0.5
        )
        assert config.max_samples == 32000
        assert config.gap_samples == 8000

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_duration_seconds", 0),
            ("max_duration_seconds", -1.0),
            ("max_samples_per_pack", 0),
            ("sample_rate", 0),
            ("gap_seconds", -0.1),
        ],
    )
    def test_validation_rejects_bad_values(self, field: str, value: float) -> None:
        kwargs: dict[str, Any] = {field: value}
        with pytest.raises(ValueError):
            PackingConfig(**kwargs)


class TestIsMainProcess:
    def test_explicit_rank(self) -> None:
        assert is_main_process(0) is True
        assert is_main_process(1) is False

    def test_auto_detect_defaults_to_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("LOCAL_RANK", "WORLD_SIZE", "SLURM_LOCALID", "SLURM_NTASKS"):
            monkeypatch.delenv(var, raising=False)
        assert get_rank_and_world_size() == (0, 1)
        assert is_main_process() is True

    def test_torchrun_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert get_rank_and_world_size() == (2, 4)
        assert is_main_process() is False

    def test_slurm_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("LOCAL_RANK", "WORLD_SIZE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SLURM_LOCALID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "2")
        assert get_rank_and_world_size() == (0, 2)
        assert is_main_process() is True
