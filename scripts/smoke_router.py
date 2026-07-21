#!/usr/bin/env python3
"""Router smoke test: end-to-end routing + wav2vec2 transcription.

Uses facebook/wav2vec2-base-960h as the merged checkpoint and the same
model as the English "expert" so only one ~380 MB download is needed.
All other languages (de, it, es, fr) have no expert checkpoint
and fall back to merged — verifying the fallback path works in practice.

Swap the expert path for a real local/HF checkpoint once available.

Run:
    uv run python scripts/smoke_router.py
    uv run python scripts/smoke_router.py --device cuda   # if GPU available
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import wave

import numpy as np
import yaml

# Unset offline flags set by pytest conftest — this script downloads real models.
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(_k, None)

import soundfile as sf  # noqa: E402

from flaime_serving.inference import (  # noqa: E402
    TranscriptionResult,
)
from flaime_serving.router import (  # noqa: E402
    EnginePool,
    LanguageNotSupportedError,
    LanguageRouter,
    RouteResult,
)


def _run_transcription_routed(
    wav_bytes: bytes,
    language_code: str,
    router: LanguageRouter,
    pool: EnginePool,
) -> tuple[TranscriptionResult, RouteResult]:
    """Inline copy of apps/demo/app.py:_run_transcription_routed for script use."""
    route = router.resolve(language_code)
    engine = pool.get_or_load(route)
    audio_array, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    result = engine.transcribe(
        audio_array, language=language_code, sample_rate=sample_rate
    )
    return result, route


TARGET_SR = 16_000

# ---------------------------------------------------------------------------
# Smoke config
# ---------------------------------------------------------------------------
# Uses wav2vec2-base-960h as both merged and English expert so only one model
# download is required.  The cache will show 1 entry (both paths are identical),
# which correctly exercises the shared-cache behaviour for same-checkpoint langs.
# ---------------------------------------------------------------------------
_SMOKE_CONFIG: dict = {
    "merged_checkpoint": {
        "path": "facebook/wav2vec2-base-960h",
        "model_type": "wav2vec2",
        "decoder": "ctc_greedy",
    },
    "languages": {
        "en": {
            "display_name": "English",
            # Same model as merged — swap for a real expert path once available.
            "expert_checkpoint": "facebook/wav2vec2-base-960h",
            "model_type": "wav2vec2",
            "decoder": "ctc_greedy",
        },
        "de": {"display_name": "German"},
        "it": {"display_name": "Italian"},
        "es": {"display_name": "Spanish"},
        "fr": {"display_name": "French"},
    },
}


def _silence_wav(duration_s: float = 1.0) -> bytes:
    """Return 1 s of silence as WAV bytes at 16 kHz (mono, int16)."""
    buf = io.BytesIO()
    n = int(TARGET_SR * duration_s)
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    return buf.getvalue()


def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  PASS: {msg}")


def main(device: str = "cpu") -> None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(_SMOKE_CONFIG, f, sort_keys=False)
        cfg_path = f.name

    print(f"Config: {cfg_path}")
    print(f"Device: {device}\n")

    router = LanguageRouter(cfg_path)
    pool = EnginePool(device=device)
    wav = _silence_wav()

    # ------------------------------------------------------------------
    # 1. Routing decisions (no model load yet)
    # ------------------------------------------------------------------
    print("=== 1. Routing decisions ===")
    for code in _SMOKE_CONFIG["languages"]:
        route = router.resolve(code)
        label = f"{route.checkpoint_type:6s}  →  {route.checkpoint_path}"
        print(f"  {code:4s}  {label}")

    route_en = router.resolve("en")
    route_de = router.resolve("de")
    _check(route_en.checkpoint_type == "expert", "en routes to expert")
    _check(route_de.checkpoint_type == "merged", "de falls back to merged")
    _check(
        route_en.checkpoint_path != route_de.checkpoint_path
        or route_en.checkpoint_path == _SMOKE_CONFIG["merged_checkpoint"]["path"],
        "fallback languages use merged path",
    )

    # ------------------------------------------------------------------
    # 2. Transcription — loads model on first call, cached thereafter
    # ------------------------------------------------------------------
    print("\n=== 2. Transcription ===")
    for code in _SMOKE_CONFIG["languages"]:
        result, route = _run_transcription_routed(wav, code, router, pool)
        print(
            f"  {code:4s}  [{route.checkpoint_type:6s}]  "
            f"text='{result.text}'  latency={result.latency_ms:.0f}ms"
        )

    # ------------------------------------------------------------------
    # 3. Cache behaviour
    # ------------------------------------------------------------------
    print("\n=== 3. Engine cache ===")
    loaded = pool.loaded_checkpoints()
    for p in loaded:
        print(f"  {p}")

    # en expert == merged path here, so cache has 1 entry; if you swap the
    # expert for a different model the count becomes 2.
    n_unique_paths = len(
        {router.resolve(c).checkpoint_path for c in _SMOKE_CONFIG["languages"]}
    )
    _check(
        len(loaded) == n_unique_paths,
        f"cache holds exactly {n_unique_paths} unique checkpoint(s)",
    )

    # ------------------------------------------------------------------
    # 4. Unsupported language rejection
    # ------------------------------------------------------------------
    print("\n=== 4. Unsupported language rejection ===")
    try:
        _run_transcription_routed(wav, "xyz", router, pool)
        _check(False, "LanguageNotSupportedError raised for 'xyz'")
    except LanguageNotSupportedError as e:
        print(f"  'xyz' rejected: {e}")
        _check(e.language_code == "xyz", "error carries rejected language code")
        _check(len(e.supported) > 0, "error carries supported language list")

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = p.parse_args()
    main(device=args.device)
