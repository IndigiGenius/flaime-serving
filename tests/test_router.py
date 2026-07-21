"""Test suite for flaime_serving.router (DEMO-03).

Run:
    uv run pytest tests/test_router.py -v

Implementation order (shortest path to GREEN):
  1. LanguageRouter._load_and_validate()  ← pure YAML + dict logic; no models
  2. LanguageRouter.supported_languages() ← trivial dict comprehension
  3. LanguageRouter.resolve()             ← routing logic; no models
  4. EnginePool.get_or_load()             ← patches ASRInferenceEngine.load()
  5. EnginePool.loaded_checkpoints()      ← reads self._cache keys
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from flaime_serving.router import (
    EnginePool,
    LanguageNotSupportedError,
    LanguageRouter,
    RouteResult,
)

# ---------------------------------------------------------------------------
# YAML fixture helpers
# ---------------------------------------------------------------------------

_MERGED = {
    "path": "ckpts/merged-v1",
    "model_type": "xeus",
    "decoder": "ctc_greedy",
}

_LANGUAGES_YAML: dict[str, Any] = {
    "merged_checkpoint": _MERGED,
    "languages": {
        "es": {
            "display_name": "Spanish",
            "expert_checkpoint": "ckpts/expert-es-v1",
            "model_type": "xeus",
            "decoder": "ctc_greedy",
        },
        "fr": {
            "display_name": "French",
            # No expert_checkpoint → falls back to merged
        },
    },
}


def _write_yaml(
    tmp_path: Path, data: dict[str, Any], filename: str = "demo_languages.yaml"
) -> Path:
    """Write a YAML dict to a temp file and return the path."""
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# TestLanguageRouterLoad — config loading and schema validation
# ---------------------------------------------------------------------------


class TestLanguageRouterLoad:
    """LanguageRouter._load_and_validate() and __init__."""

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        """A well-formed YAML file produces a usable LanguageRouter."""
        cfg = _write_yaml(tmp_path, _LANGUAGES_YAML)
        router = LanguageRouter(cfg)
        assert router is not None

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent config path raises FileNotFoundError at construction."""
        with pytest.raises(FileNotFoundError, match="Serving config not found"):
            LanguageRouter(tmp_path / "nonexistent.yaml")

    def test_missing_merged_checkpoint_key_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """YAML missing 'merged_checkpoint' raises ValueError naming the key."""
        bad = {k: v for k, v in _LANGUAGES_YAML.items() if k != "merged_checkpoint"}
        cfg = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="merged_checkpoint"):
            LanguageRouter(cfg)

    def test_missing_languages_key_raises_value_error(self, tmp_path: Path) -> None:
        """YAML missing 'languages' raises ValueError naming the key."""
        bad = {k: v for k, v in _LANGUAGES_YAML.items() if k != "languages"}
        cfg = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="languages"):
            LanguageRouter(cfg)

    def test_merged_checkpoint_missing_path_key_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        """merged_checkpoint missing 'path' sub-key raises an actionable ValueError."""
        bad = {
            **_LANGUAGES_YAML,
            "merged_checkpoint": {"model_type": "xeus", "decoder": "ctc_greedy"},
        }
        cfg = _write_yaml(tmp_path, bad)
        with pytest.raises(ValueError, match="path"):
            LanguageRouter(cfg)

    def test_bad_yaml_syntax_raises_value_error(self, tmp_path: Path) -> None:
        """A syntactically invalid YAML file raises a ValueError with the file path."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("merged_checkpoint: {\n  path: [unclosed")
        # TODO(DEMO-03): Decide whether to wrap yaml.YAMLError in ValueError or let it
        #   propagate.  Either is acceptable; update this assertion to match the
        #   chosen approach.  The message must include the file path.
        with pytest.raises((ValueError, Exception), match=str(bad_yaml)):
            LanguageRouter(bad_yaml)


# ---------------------------------------------------------------------------
# TestLanguageRouterResolve — expert-preferred, merged-fallback, rejection
# ---------------------------------------------------------------------------


class TestLanguageRouterResolve:
    """LanguageRouter.resolve() routing logic."""

    @pytest.fixture()
    def router(self, tmp_path: Path) -> LanguageRouter:
        cfg = _write_yaml(tmp_path, _LANGUAGES_YAML)
        return LanguageRouter(cfg)

    def test_expert_preferred_when_expert_checkpoint_present(
        self, router: LanguageRouter
    ) -> None:
        """Language with expert_checkpoint resolves to checkpoint_type='expert'."""
        result = router.resolve("es")
        assert result.checkpoint_type == "expert"
        assert result.checkpoint_path == "ckpts/expert-es-v1"
        assert result.language_code == "es"
        assert result.display_name == "Spanish"

    def test_merged_fallback_when_no_expert_checkpoint(
        self, router: LanguageRouter
    ) -> None:
        """Language without expert_checkpoint resolves to checkpoint_type='merged'."""
        result = router.resolve("fr")
        assert result.checkpoint_type == "merged"
        assert result.checkpoint_path == _MERGED["path"]
        assert result.language_code == "fr"
        assert result.display_name == "French"

    def test_unsupported_language_raises_language_not_supported_error(
        self, router: LanguageRouter
    ) -> None:
        """Language code not in config raises LanguageNotSupportedError (no silent fallback)."""
        with pytest.raises(LanguageNotSupportedError) as exc_info:
            router.resolve("xyz")
        err = exc_info.value
        assert err.language_code == "xyz"
        assert "es" in err.supported
        assert "fr" in err.supported

    def test_resolve_uses_per_language_model_type_when_specified(
        self, tmp_path: Path
    ) -> None:
        """Per-language model_type overrides merged_checkpoint model_type."""
        data: dict[str, Any] = {
            **_LANGUAGES_YAML,
            "languages": {
                "es": {
                    "display_name": "Spanish",
                    "expert_checkpoint": "ckpts/expert-es-v1",
                    "model_type": "whisper",  # different from merged "xeus"
                    "decoder": "ctc_greedy",
                },
            },
        }
        cfg = _write_yaml(tmp_path, data)
        router = LanguageRouter(cfg)
        result = router.resolve("es")
        assert result.model_type == "whisper"

    def test_resolve_inherits_merged_model_type_when_not_specified_per_language(
        self, tmp_path: Path
    ) -> None:
        """Language entry without model_type inherits merged_checkpoint's model_type."""
        data: dict[str, Any] = {
            **_LANGUAGES_YAML,
            "languages": {
                "fr": {
                    "display_name": "French",
                    # No model_type — should inherit from merged_checkpoint
                },
            },
        }
        cfg = _write_yaml(tmp_path, data)
        router = LanguageRouter(cfg)
        result = router.resolve("fr")
        assert result.model_type == _MERGED["model_type"]


# ---------------------------------------------------------------------------
# TestLanguageRouterSupportedLanguages
# ---------------------------------------------------------------------------


class TestLanguageRouterSupportedLanguages:
    """LanguageRouter.supported_languages() returns config-parity mapping."""

    @pytest.fixture()
    def router(self, tmp_path: Path) -> LanguageRouter:
        cfg = _write_yaml(tmp_path, _LANGUAGES_YAML)
        return LanguageRouter(cfg)

    def test_returns_all_configured_languages(self, router: LanguageRouter) -> None:
        langs = router.supported_languages()
        assert set(langs.keys()) == {"es", "fr"}

    def test_display_names_match_config(self, router: LanguageRouter) -> None:
        langs = router.supported_languages()
        assert langs["es"] == "Spanish"
        assert langs["fr"] == "French"

    def test_returns_dict_not_list(self, router: LanguageRouter) -> None:
        langs = router.supported_languages()
        assert isinstance(langs, dict)


# ---------------------------------------------------------------------------
# TestEnginePool — cache hit, cache miss, and on-miss loading
# ---------------------------------------------------------------------------


def _make_stub_route(
    checkpoint_type: str = "expert",
    checkpoint_path: str = "ckpts/expert-es-v1",
    model_type: str = "xeus",
    decoder: str = "ctc_greedy",
    language_code: str = "es",
    display_name: str = "Spanish",
) -> RouteResult:
    return RouteResult(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        decoder=decoder,
        checkpoint_type=checkpoint_type,  # type: ignore[arg-type]
        language_code=language_code,
        display_name=display_name,
    )


class TestEnginePool:
    """EnginePool caches engines per checkpoint_path."""

    def test_get_or_load_calls_inference_engine_on_first_load(self) -> None:
        """Cache miss: ASRInferenceEngine.load() is called exactly once."""
        stub_engine = MagicMock()
        route = _make_stub_route()

        with patch(
            "flaime_serving.router.ASRInferenceEngine.load", return_value=stub_engine
        ) as mock_load:
            pool = EnginePool(device="cpu")
            engine = pool.get_or_load(route)

        mock_load.assert_called_once_with(
            route.checkpoint_path,
            model_type=route.model_type,
            device="cpu",
            decoder=route.decoder,
            warmup=False,
        )
        assert engine is stub_engine

    def test_engine_cache_hit_does_not_reload_from_disk(self) -> None:
        """Cache hit: ASRInferenceEngine.load() is called at most once for a given path."""
        stub_engine = MagicMock()
        stub_engine.decoder = "ctc_greedy"
        route = _make_stub_route()

        with patch(
            "flaime_serving.router.ASRInferenceEngine.load", return_value=stub_engine
        ) as mock_load:
            pool = EnginePool(device="cpu")
            engine_first = pool.get_or_load(route)
            engine_second = pool.get_or_load(route)

        assert mock_load.call_count == 1, "Engine must not be re-loaded on cache hit"
        assert engine_first is engine_second

    def test_different_checkpoints_produce_different_engines(self) -> None:
        """Two distinct checkpoint paths produce two distinct engine instances."""
        stub_es = MagicMock(name="engine_es")
        stub_fr = MagicMock(name="engine_fr")
        route_es = _make_stub_route(checkpoint_path="ckpts/es", language_code="es")
        route_fr = _make_stub_route(
            checkpoint_path="ckpts/fr",
            language_code="fr",
            checkpoint_type="merged",
        )

        def _fake_load(path: str, **_: object) -> MagicMock:
            return stub_es if path == "ckpts/es" else stub_fr

        with patch(
            "flaime_serving.router.ASRInferenceEngine.load", side_effect=_fake_load
        ):
            pool = EnginePool(device="cpu")
            e_es = pool.get_or_load(route_es)
            e_fr = pool.get_or_load(route_fr)

        assert e_es is stub_es
        assert e_fr is stub_fr

    def test_loaded_checkpoints_returns_sorted_paths(self) -> None:
        """loaded_checkpoints() returns a sorted list of loaded paths."""
        stub_engine = MagicMock()
        route_a = _make_stub_route(checkpoint_path="ckpts/z", language_code="z")
        route_b = _make_stub_route(checkpoint_path="ckpts/a", language_code="a")

        with patch(
            "flaime_serving.router.ASRInferenceEngine.load", return_value=stub_engine
        ):
            pool = EnginePool(device="cpu")
            pool.get_or_load(route_a)
            pool.get_or_load(route_b)

        assert pool.loaded_checkpoints() == ["ckpts/a", "ckpts/z"]

    def test_evict_removes_entry_from_cache(self) -> None:
        """evict() removes the given checkpoint from the pool and returns True."""
        stub_engine = MagicMock()
        route = _make_stub_route()

        with patch(
            "flaime_serving.router.ASRInferenceEngine.load", return_value=stub_engine
        ):
            pool = EnginePool(device="cpu")
            pool.get_or_load(route)
            removed = pool.evict(route.checkpoint_path)

        assert removed is True
        assert route.checkpoint_path not in pool.loaded_checkpoints()

    def test_evict_returns_false_for_unknown_path(self) -> None:
        """evict() returns False when the path is not in the cache."""
        pool = EnginePool(device="cpu")
        assert pool.evict("ckpts/nonexistent") is False
