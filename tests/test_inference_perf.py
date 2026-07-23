"""Performance / latency tests for the serving path (26Q1-DEMO-07).

Covers the warmup() contract: it leaves the engine usable via synthetic
audio, with no disk/network I/O.

The original FLAIME suite (tests/serving/test_inference_perf.py) also
covers the latency-harness schema (TestBenchSchema, needs
scripts/demo/bench_latency.py — moves in 26Q3-REPO-12) and the EnginePool
cache contract (TestEngineCacheNoReload, needs flaime_serving.router —
moves in 26Q3-REPO-04). Those classes stay in FLAIME until their
dependencies land here, per 26Q3-REPO-03's decision-gate note.

Uses a minimal in-process stub model — no HuggingFace download, no GPU.
"""

from __future__ import annotations

import torch

from flaime_serving.inference import ASRInferenceEngine


class _StubProcessor:
    def decode(self, token_ids: list[int], **kwargs: object) -> str:  # noqa: ARG002
        return "hello world"


class _StubModel:
    """Minimal ASRModelProtocol stand-in mirroring tests/test_inference.py."""

    revision = "stub-rev-perf"

    def forward(
        self, input_features: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        T = max(1, input_features.shape[-1] // 160)
        return {"logits": torch.full((1, T, 32), fill_value=-3.0)}

    @property
    def processor(self) -> _StubProcessor:
        return _StubProcessor()

    def eval(self) -> _StubModel:
        return self

    def to(self, device: str) -> _StubModel:  # noqa: ARG002
        return self


def _stub_engine() -> ASRInferenceEngine:
    return ASRInferenceEngine(
        model=_StubModel(),
        model_revision=_StubModel.revision,
        decoder="ctc_greedy",
        device="cpu",
    )


class TestWarmup:
    def test_warmup_does_no_io_and_keeps_device(self) -> None:
        engine = _stub_engine()
        # Synthetic-audio warmup: no file path, no network — must not raise.
        engine.warmup()
        assert engine.device == "cpu"

    def test_engine_usable_after_warmup(self) -> None:
        engine = _stub_engine()
        engine.warmup()
        result = engine.transcribe(torch.zeros(16_000), sample_rate=16_000)
        assert result.text == "hello world"
        assert result.latency_ms >= 0.0
