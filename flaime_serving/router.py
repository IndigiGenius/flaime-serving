"""Language-to-checkpoint router for the FLAIME demo serving pipeline (DEMO-03).

Routing rules (applied in priority order):
  1. expert   — use per-language expert checkpoint when one is configured.
  2. merged   — fall back to the shared merged checkpoint otherwise.
  3. reject   — raise LanguageNotSupportedError for any code not in the config.
                No silent fallback — community partners must see a clear message.

Typical usage::

    router = LanguageRouter.from_yaml("configs/serving/demo_languages.yaml")
    pool = EnginePool(device="cpu")

    route = router.resolve("eng")           # RouteResult(checkpoint_type="expert", …)
    engine = pool.get_or_load(route)         # cached; only loads from disk once
    result = engine.transcribe(audio, language="eng")
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal

import yaml

from flaime_serving.inference import ASRInferenceEngine

# ---------------------------------------------------------------------------
# Internal config types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _MergedConfig:
    path: str
    model_type: str
    decoder: str


@dataclasses.dataclass(frozen=True)
class _LanguageEntry:
    display_name: str
    expert_checkpoint: str | None = None
    model_type: str | None = None
    decoder: str | None = None
    status: str | None = None  # e.g. "transfer_eval_pending" for held-out langs


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RouteResult:
    """Resolved routing information for a single language request.

    Attributes:
        checkpoint_path: Path (or HF Hub ID) of the checkpoint to load.
        model_type: Architecture key understood by ASRModelFactory.
        decoder: Decoding strategy string (e.g. ``"ctc_greedy"``).
        checkpoint_type: ``"expert"`` when a per-language expert checkpoint is
            used; ``"merged"`` when the shared merged checkpoint is used.
        language_code: BCP-47 / ISO 639-3 code passed to ``resolve()``.
        display_name: Human-readable language name from the config.
    """

    checkpoint_path: str
    model_type: str
    decoder: str
    checkpoint_type: Literal["expert", "merged"]
    language_code: str
    display_name: str


class LanguageNotSupportedError(ValueError):
    """Raised by LanguageRouter.resolve() for an unsupported language code.

    Carries both the rejected code and the list of supported codes so callers
    can surface an actionable error message without additional lookups.
    """

    def __init__(self, language_code: str, supported: list[str]) -> None:
        self.language_code = language_code
        self.supported = supported
        super().__init__(
            f"Language {language_code!r} is not supported by this demo configuration. "
            f"Supported languages: {supported}"
        )


# ---------------------------------------------------------------------------
# LanguageRouter
# ---------------------------------------------------------------------------


class LanguageRouter:
    """Maps a BCP-47/ISO 639-3 language code to a checkpoint RouteResult.

    Loads its full routing table from a single YAML config on construction.
    The config schema is documented in ``configs/serving/demo_languages.yaml``.

    Raises ``ValueError`` with an actionable message at construction time if the
    YAML is missing required keys — fail fast so startup errors are obvious.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Load and validate the routing config.

        Args:
            config_path: Path to the YAML routing config
                (e.g. ``configs/serving/demo_languages.yaml``).

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: If the YAML is missing ``merged_checkpoint`` or
                ``languages`` keys, or if ``merged_checkpoint`` is missing
                required sub-keys.
        """
        self._config_path = Path(config_path)
        self._merged, self._languages = self._load_and_validate(self._config_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LanguageRouter:
        """Convenience constructor; identical to ``__init__`` with a Path.

        Args:
            path: Path to the YAML routing config.

        Returns:
            Initialised LanguageRouter.
        """
        return cls(path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve(self, language_code: str) -> RouteResult:
        """Return the RouteResult for *language_code*.

        Routing priority:
          1. expert   — entry has ``expert_checkpoint`` key.
          2. merged   — entry exists but has no ``expert_checkpoint``.
          3. reject   — entry does not exist → LanguageNotSupportedError.

        Args:
            language_code: BCP-47 / ISO 639-3 code (e.g. ``"eng"``).

        Returns:
            RouteResult with ``checkpoint_type`` set to ``"expert"`` or
            ``"merged"``.

        Raises:
            LanguageNotSupportedError: If language_code is not in the config.
        """
        if language_code not in self._languages:
            raise LanguageNotSupportedError(language_code, list(self._languages))

        entry = self._languages[language_code]

        if entry.expert_checkpoint is not None:
            return RouteResult(
                checkpoint_path=entry.expert_checkpoint,
                model_type=entry.model_type or self._merged.model_type,
                decoder=entry.decoder or self._merged.decoder,
                checkpoint_type="expert",
                language_code=language_code,
                display_name=entry.display_name,
            )

        return RouteResult(
            checkpoint_path=self._merged.path,
            model_type=entry.model_type or self._merged.model_type,
            decoder=entry.decoder or self._merged.decoder,
            checkpoint_type="merged",
            language_code=language_code,
            display_name=entry.display_name,
        )

    def supported_languages(self) -> dict[str, str]:
        """Return a mapping of language code → display name for all configured languages.

        Used by the Streamlit UI to populate the language dropdown.

        Returns:
            Ordered dict ``{code: display_name}`` in config-file order.
        """
        return {code: entry.display_name for code, entry in self._languages.items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_validate(
        config_path: Path,
    ) -> tuple[_MergedConfig, dict[str, _LanguageEntry]]:
        """Load YAML from config_path and validate top-level schema.

        Args:
            config_path: Path to the YAML file.

        Returns:
            Tuple of (merged config, per-language entries) with all fields typed.

        Raises:
            FileNotFoundError: If config_path does not exist.
            ValueError: On YAML parse error or missing required keys, with a
                message naming the missing key and the file path so the operator
                knows exactly what to fix.
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Serving config not found: {config_path}. "
                "Create it from configs/serving/demo_languages.yaml."
            )

        try:
            raw_obj = yaml.safe_load(config_path.read_text())
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        if not isinstance(raw_obj, dict):
            raise ValueError(f"Expected a YAML mapping at top level in {config_path}")

        raw: dict[str, Any] = raw_obj

        merged_raw = raw.get("merged_checkpoint")
        if not isinstance(merged_raw, dict):
            raise ValueError(
                f"Missing required key 'merged_checkpoint' in {config_path}"
            )
        for sub in ("path", "model_type", "decoder"):
            if sub not in merged_raw:
                raise ValueError(
                    f"merged_checkpoint missing required key '{sub}' in {config_path}"
                )

        languages_raw = raw.get("languages")
        if not isinstance(languages_raw, dict) or not languages_raw:
            raise ValueError(
                f"Missing or empty required key 'languages' in {config_path}"
            )
        for code, entry in languages_raw.items():
            if not isinstance(entry, dict) or "display_name" not in entry:
                raise ValueError(
                    f"Language '{code}' missing required key 'display_name' in {config_path}"
                )
        merged = _MergedConfig(
            path=raw["merged_checkpoint"]["path"],
            model_type=raw["merged_checkpoint"]["model_type"],
            decoder=raw["merged_checkpoint"]["decoder"],
        )
        languages = {
            code: _LanguageEntry(
                display_name=entry["display_name"],
                expert_checkpoint=entry.get("expert_checkpoint"),
                model_type=entry.get("model_type"),
                decoder=entry.get("decoder"),
                status=entry.get("status"),
            )
            for code, entry in raw["languages"].items()
        }
        return merged, languages


# ---------------------------------------------------------------------------
# EnginePool
# ---------------------------------------------------------------------------


class EnginePool:
    """Cache of ASRInferenceEngine instances keyed by checkpoint path.

    Ensures a given checkpoint is loaded from disk at most once per process,
    so switching the language selector in the UI does not trigger a slow
    model reload on every request.
    """

    def __init__(self, device: str | None = None, warmup: bool = False) -> None:
        """Initialise an empty pool.

        Args:
            device: Torch device string (``"cpu"``, ``"cuda"``, …).  ``None``
                delegates auto-detection to ASRInferenceEngine.load().
            warmup: When ``True``, each engine runs one throwaway inference as
                it is loaded (DEMO-07), so the one-time cold-start cost is paid
                at load time rather than on the first user-facing transcription.
                Only worthwhile when ``get_or_load`` is called *ahead* of the
                request (e.g. the demo warms on language-select); warming inside
                the request that triggers the load just adds a forward pass.
                Defaults to ``False`` to preserve lazy-load latency.
        """
        self._device = device
        self._warmup = warmup
        self._cache: dict[str, ASRInferenceEngine] = {}

    def get_or_load(self, route: RouteResult) -> ASRInferenceEngine:
        """Return a cached engine for *route*, loading from disk if necessary.

        Thread-safety: not guaranteed.  For the single-process Streamlit demo
        this is fine; add a lock if the serving layer is ever made multi-threaded.

        Args:
            route: RouteResult from LanguageRouter.resolve().

        Returns:
            ASRInferenceEngine ready for inference.

        Raises:
            FileNotFoundError: Propagated from ASRInferenceEngine.load() if the
                checkpoint path does not exist on disk.
        """
        key = route.checkpoint_path
        cached = self._cache.get(key)
        if cached is None:
            cached = ASRInferenceEngine.load(
                route.checkpoint_path,
                model_type=route.model_type,
                device=self._device,
                decoder=route.decoder,
                warmup=self._warmup,
            )
            self._cache[key] = cached
        elif cached.decoder != route.decoder:
            raise ValueError(
                f"EnginePool cache collision for {key!r}: cached decoder {cached.decoder!r} "
                f"!= requested {route.decoder!r}. Use distinct checkpoint paths per decoder."
            )
        return cached

    def loaded_checkpoints(self) -> list[str]:
        """Return a sorted list of checkpoint paths currently held in the cache.

        Useful for introspection and tests.

        Returns:
            Sorted list of checkpoint path strings.
        """
        return sorted(self._cache.keys())

    def evict(self, checkpoint_path: str) -> bool:
        """Remove a checkpoint from the cache (frees GPU memory on eviction).

        Args:
            checkpoint_path: Key to remove.

        Returns:
            ``True`` if the entry existed and was removed; ``False`` otherwise.
        """
        if checkpoint_path in self._cache:
            del self._cache[checkpoint_path]
            return True
        return False
