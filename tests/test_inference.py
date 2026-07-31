"""Test suite for flaime_serving.inference (DEMO-01).

Run: uv run pytest tests/test_inference.py -v

CLI coverage (TestCLIArgparse, TestCLIRunTranscribe against
flaime.cli.commands.serve) stays in FLAIME's copy of this file — that module
moves to flaime_serving.cli in 26Q3-REPO-06, and its tests move with it.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 16_000  # must stay in sync with flaime_serving.inference.TARGET_SR

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_wav_file(
    path: Path,
    num_samples: int = TARGET_SR,
    sample_rate: int = TARGET_SR,
    num_channels: int = 1,
) -> Path:
    """Write a minimal valid WAV file (silence) and return the path."""
    samples = np.zeros(num_samples * num_channels, dtype=np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return path


class _StubProcessor:
    """Minimal stand-in for a model processor."""

    def decode(self, token_ids: list[int], **kwargs: object) -> str:  # noqa: ARG002
        return "hello world"


class _StubModel:
    """Minimal stand-in for a loaded ASR model.

    Mirrors the BaseASRModel / ASRModelProtocol inference interface so
    transcribe() tests run without HuggingFace downloads or GPU:
      forward()  → {"logits": Tensor(1, T_out, vocab_size)}
      processor  → object with decode(token_ids) → str
      eval()     → self
      to(device) → self
    """

    revision = "stub-rev-abc123"

    def forward(
        self, input_features: torch.Tensor, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Return fake CTC logits shaped (1, T_out, vocab_size)."""
        T = max(1, input_features.shape[-1] // 160)
        return {"logits": torch.full((1, T, 32), fill_value=-3.0)}

    @property
    def processor(self) -> _StubProcessor:
        return _StubProcessor()

    def eval(self) -> _StubModel:  # noqa: ARG002
        return self

    def to(self, device: str) -> _StubModel:  # noqa: ARG002
        return self


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_result():
    """Pre-built TranscriptionResult for serialisation tests."""
    from flaime_serving.inference import TranscriptionResult

    return TranscriptionResult(
        text="hello world",
        language="en",
        confidence=0.95,
        latency_ms=42.0,
        model_revision="stub-rev-abc123",
        decoder="ctc_greedy",
    )


@pytest.fixture()
def stub_engine():
    """ASRInferenceEngine backed by _StubModel — no real checkpoint."""
    from flaime_serving.inference import ASRInferenceEngine

    return ASRInferenceEngine(
        model=_StubModel(),
        model_revision=_StubModel.revision,
        decoder="ctc_greedy",
        device="cpu",
    )


@pytest.fixture()
def wav_path(tmp_path) -> Path:
    """16 kHz mono WAV with 1 s of silence."""
    return _make_wav_file(tmp_path / "test.wav")


@pytest.fixture()
def wav_path_8k(tmp_path) -> Path:
    """8 kHz mono WAV with 1 s of silence — for resampling tests."""
    return _make_wav_file(tmp_path / "test_8k.wav", num_samples=8000, sample_rate=8000)


@pytest.fixture()
def wav_path_44k(tmp_path) -> Path:
    """44.1 kHz mono WAV with 1 s of silence — exercises the canonical
    consumer-audio → ASR-target downsample path called out in the spec."""
    return _make_wav_file(
        tmp_path / "test_44k.wav", num_samples=44_100, sample_rate=44_100
    )


@pytest.fixture()
def sine_wav_path(tmp_path) -> Path:
    """440 Hz sine tone at 16 kHz, 0.5 s, stored as int16 — non-trivial audio for parity tests."""
    duration_s = 0.5
    n = int(duration_s * TARGET_SR)
    t = np.linspace(0, duration_s, n, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    path = tmp_path / "sine.wav"
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(samples.tobytes())
    return path


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_all_ac_fields_present(self, stub_result) -> None:
        for field in (
            "text",
            "language",
            "confidence",
            "latency_ms",
            "model_revision",
            "decoder",
        ):
            assert hasattr(stub_result, field), f"missing field: {field}"

    def test_to_json_produces_valid_json(self, stub_result) -> None:
        j = stub_result.to_json()
        parsed = json.loads(j)
        assert parsed["text"] == "hello world"
        assert parsed["language"] == "en"
        assert parsed["decoder"] == "ctc_greedy"

    def test_to_json_contains_all_fields(self, stub_result) -> None:
        data = json.loads(stub_result.to_json())
        required = {
            "text",
            "language",
            "confidence",
            "latency_ms",
            "model_revision",
            "decoder",
        }
        assert required.issubset(data.keys())

    def test_from_json_round_trips(self, stub_result) -> None:
        from flaime_serving.inference import TranscriptionResult

        restored = TranscriptionResult.from_json(stub_result.to_json())
        assert restored == stub_result

    def test_from_json_field_values(self, stub_result) -> None:
        from flaime_serving.inference import TranscriptionResult

        r = TranscriptionResult.from_json(stub_result.to_json())
        assert r.confidence == pytest.approx(0.95)
        assert r.latency_ms == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# _normalize_audio
# ---------------------------------------------------------------------------


class TestNormalizeAudio:
    def test_wav_path_str_returns_float32_array(self, wav_path) -> None:
        from flaime_serving.inference import _normalize_audio

        result = _normalize_audio(str(wav_path))
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_wav_path_pathlib_returns_array(self, wav_path) -> None:
        from flaime_serving.inference import _normalize_audio

        result = _normalize_audio(wav_path)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_numpy_input_returns_float32_array(self) -> None:
        from flaime_serving.inference import _normalize_audio

        arr = np.zeros(TARGET_SR, dtype=np.float32)
        result = _normalize_audio(arr, sample_rate=TARGET_SR)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_tensor_input_returns_float32_array(self) -> None:
        from flaime_serving.inference import _normalize_audio

        t = torch.zeros(TARGET_SR)
        result = _normalize_audio(t, sample_rate=TARGET_SR)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_resampling_8k_to_16k_doubles_length(self, wav_path_8k) -> None:
        """8 000 samples @8 kHz → ~16 000 samples @16 kHz."""
        from flaime_serving.inference import _normalize_audio

        result = _normalize_audio(wav_path_8k)
        assert abs(len(result) - TARGET_SR) < 100

    def test_resampling_44k_to_16k_yields_correct_length(self, wav_path_44k) -> None:
        """44 100 samples @44.1 kHz → ~16 000 samples @16 kHz.

        Spec calls out 44.1k → 16k explicitly — the consumer-audio path
        most likely to arrive at the demo from a phone or laptop mic.
        """
        from flaime_serving.inference import _normalize_audio

        result = _normalize_audio(wav_path_44k)
        # Resampling is fractional (16000/44100 ≈ 0.3628); torchaudio's
        # implementation introduces small length drift, so a tolerance band
        # is correct here.
        assert abs(len(result) - TARGET_SR) < 100

    def test_stereo_wav_collapsed_to_mono(self, tmp_path) -> None:
        stereo_wav = _make_wav_file(tmp_path / "stereo.wav", num_channels=2)
        from flaime_serving.inference import _normalize_audio

        result = _normalize_audio(stereo_wav)
        assert result.ndim == 1

    def test_stereo_numpy_ct_layout_collapsed_to_mono(self) -> None:
        """(C, T) layout from torchaudio: channels first."""
        from flaime_serving.inference import _normalize_audio

        arr = np.zeros((2, TARGET_SR), dtype=np.float32)  # (C=2, T)
        result = _normalize_audio(arr, sample_rate=TARGET_SR)
        assert result.ndim == 1
        assert len(result) == TARGET_SR

    def test_stereo_numpy_tc_layout_collapsed_to_mono(self) -> None:
        """(T, C) layout from soundfile: time first."""
        from flaime_serving.inference import _normalize_audio

        arr = np.zeros((TARGET_SR, 2), dtype=np.float32)  # (T, C=2)
        result = _normalize_audio(arr, sample_rate=TARGET_SR)
        assert result.ndim == 1
        assert len(result) == TARGET_SR

    def test_stereo_tensor_tc_layout_collapsed_to_mono(self) -> None:
        """(T, C) Tensor layout also handled correctly."""
        from flaime_serving.inference import _normalize_audio

        t = torch.zeros(TARGET_SR, 2)  # (T, C=2)
        result = _normalize_audio(t, sample_rate=TARGET_SR)
        assert result.ndim == 1
        assert len(result) == TARGET_SR

    def test_int16_numpy_scaled_to_unit_range(self) -> None:
        """int16 PCM in [-32768, 32767] must land in [-1, 1] after normalisation."""
        from flaime_serving.inference import _normalize_audio

        arr = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        result = _normalize_audio(arr, sample_rate=TARGET_SR)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0
        assert abs(result[1] - 0.5) < 0.01  # 16384 / 32767 ≈ 0.5

    def test_int16_tensor_scaled_to_unit_range(self) -> None:
        """int16 Tensor PCM must be scaled the same way as int16 ndarray."""
        from flaime_serving.inference import _normalize_audio

        t = torch.tensor([0, 16384, -16384, 32767], dtype=torch.int16)
        result = _normalize_audio(t, sample_rate=TARGET_SR)
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_int16_numpy_and_tensor_parity(self) -> None:
        """Same int16 samples via ndarray and Tensor must produce identical output."""
        from flaime_serving.inference import _normalize_audio

        samples = np.array([0, 8192, -8192, 32767], dtype=np.int16)
        r_np = _normalize_audio(samples, sample_rate=TARGET_SR)
        r_t = _normalize_audio(torch.from_numpy(samples), sample_rate=TARGET_SR)
        np.testing.assert_allclose(r_np, r_t, rtol=1e-5)

    def test_missing_file_raises_file_not_found(self) -> None:
        from flaime_serving.inference import _normalize_audio

        with pytest.raises(FileNotFoundError):
            _normalize_audio("/nonexistent/does_not_exist.wav")

    def test_array_without_sample_rate_raises_value_error(self) -> None:
        from flaime_serving.inference import _normalize_audio

        with pytest.raises(ValueError, match="sample_rate"):
            _normalize_audio(np.zeros(100, dtype=np.float32))

    def test_tensor_without_sample_rate_raises_value_error(self) -> None:
        from flaime_serving.inference import _normalize_audio

        with pytest.raises(ValueError, match="sample_rate"):
            _normalize_audio(torch.zeros(100))

    def test_resample_warns_when_input_rate_differs(self, wav_path_8k) -> None:
        """Spec: warn on resample so callers know the audio was modified.

        Uses manual ``catch_warnings`` rather than ``pytest.warns`` because
        the shared module-level ``__warningregistry__`` for the source line
        in ``_normalize_audio`` gets populated by earlier resample tests in
        the same class, and pytest.warns does not always reset that registry
        — leading to flaky "DID NOT WARN" failures when run as part of the
        full suite (passes in isolation).
        """
        import warnings as _warnings

        from flaime_serving.inference import _normalize_audio

        with _warnings.catch_warnings(record=True) as captured:
            _warnings.simplefilter("always")
            _normalize_audio(wav_path_8k)

        resample_warnings = [
            w
            for w in captured
            if issubclass(w.category, UserWarning)
            and "resampl" in str(w.message).lower()
        ]
        assert resample_warnings, (
            f"Expected a resample UserWarning; got: {[str(w.message) for w in captured]}"
        )

    def test_no_warning_when_rate_already_matches_target(self, wav_path) -> None:
        """Sanity: *our* resample warning does not fire when input is already 16 kHz.

        We can't promote *all* warnings to errors here: torchaudio 2.8 entered
        maintenance mode and ``torchaudio.load`` now emits its own library-level
        deprecation warnings (torchcodec migration, torio decoder) that are
        outside our control and globally filtered in pyproject. So we scope the
        check the same way the sibling ``test_resample_warns_when_input_rate_differs``
        does — by message content — and assert no resample warning is present.
        (A filename-based scope would never match: ``_normalize_audio`` emits its
        warning with ``stacklevel=2``, so Python attributes it to the caller, not
        to ``serving/inference.py``.)
        """
        import warnings

        from flaime_serving.inference import _normalize_audio

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _normalize_audio(wav_path)

        resample_warnings = [
            w
            for w in captured
            if issubclass(w.category, UserWarning)
            and "resampl" in str(w.message).lower()
        ]
        assert not resample_warnings, (
            "Expected no resample warning on the no-resample path; "
            f"got: {[str(w.message) for w in resample_warnings]}"
        )

    def test_invalid_sample_rate_raises_value_error(self) -> None:
        """Spec: reject sample rates the resampler can't handle."""
        from flaime_serving.inference import _normalize_audio

        with pytest.raises(ValueError, match="sample_rate"):
            _normalize_audio(np.zeros(100, dtype=np.float32), sample_rate=0)
        with pytest.raises(ValueError, match="sample_rate"):
            _normalize_audio(np.zeros(100, dtype=np.float32), sample_rate=-16000)


# ---------------------------------------------------------------------------
# ASRInferenceEngine.load()
# ---------------------------------------------------------------------------


class TestASRInferenceEngineLoad:
    def test_missing_checkpoint_raises_file_not_found(self, tmp_path) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        with pytest.raises(FileNotFoundError):
            ASRInferenceEngine.load(tmp_path / "nonexistent", model_type="xeus")

    def test_load_returns_engine_instance(self, tmp_path, mocker) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text('{"model_revision": "abc123"}')

        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus", device="cpu")
        assert isinstance(engine, ASRInferenceEngine)

    def test_load_stores_decoder_attribute(self, tmp_path, mocker) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")

        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus", decoder="ctc_beam5")
        assert engine.decoder == "ctc_beam5"

    def test_load_reads_model_revision_from_config(self, tmp_path, mocker) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text('{"model_revision": "v42"}')

        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus")
        assert engine.model_revision == "v42"

    def test_load_invalid_decoder_raises_value_error(self, tmp_path) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        with pytest.raises(ValueError, match="decoder"):
            ASRInferenceEngine.load(ckpt, model_type="xeus", decoder="beam_search")

    def test_direct_constructor_invalid_decoder_raises_value_error(self) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        with pytest.raises(ValueError, match="decoder"):
            ASRInferenceEngine(
                model=_StubModel(),
                model_revision="rev",
                decoder="beam_search",
                device="cpu",
            )

    def test_load_device_default_is_cuda_when_available(self, tmp_path, mocker) -> None:
        """Spec: device default auto-detects CUDA; CPU only when no GPU."""
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()
        mocker.patch("torch.cuda.is_available", return_value=True)

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus")
        assert engine.device == "cuda"

    def test_load_device_default_is_cpu_when_no_cuda(self, tmp_path, mocker) -> None:
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()
        mocker.patch("torch.cuda.is_available", return_value=False)

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus")
        assert engine.device == "cpu"

    def test_load_explicit_device_overrides_auto_detect(self, tmp_path, mocker) -> None:
        """Caller passing device=... must win over auto-detect."""
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()
        mocker.patch("torch.cuda.is_available", return_value=True)

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus", device="cpu")
        assert engine.device == "cpu"

    def test_load_hf_hub_id_skips_existence_check(self, mocker) -> None:
        """A 'org/model' Hub ID must bypass the local-path existence check.

        Requires the explicit remote opt-in: the ID resolves over the network,
        so refusing by default is what makes an offline deployment offline.
        """
        from flaime_serving.inference import ASRInferenceEngine

        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()

        # Must not raise FileNotFoundError even though the path doesn't exist locally.
        engine = ASRInferenceEngine.load(
            "facebook/wav2vec2-base-960h",
            model_type="wav2vec2",
            device="cpu",
            allow_remote=True,
        )
        assert isinstance(engine, ASRInferenceEngine)
        # model_revision falls back to Path.name when no local config.json
        assert engine.model_revision == "wav2vec2-base-960h"

    def test_load_refuses_a_hub_id_by_default(self) -> None:
        """Remote resolution is opt-in: the default must not reach the network.

        Without this, a routing-YAML entry of "facebook/wav2vec2-base" silently
        downloads at load time on a deployment whose premise is offline
        operation — and fails slowly (network timeout) rather than clearly.
        """
        from flaime_serving.inference import (
            ASRInferenceEngine,
            RemoteCheckpointRefused,
        )

        with pytest.raises(RemoteCheckpointRefused) as excinfo:
            ASRInferenceEngine.load("facebook/wav2vec2-base", model_type="wav2vec2")

        message = str(excinfo.value)
        assert "facebook/wav2vec2-base" in message, "name the offending value"
        assert "allow_remote" in message, "point at the opt-in"

    def test_remote_refusal_stays_a_file_not_found_error(self) -> None:
        """The refusal must remain a FileNotFoundError for downstream consumers.

        flaime-demo's errors.py maps FileNotFoundError to "no model is loaded";
        a bare ValueError would surface as "couldn't read that audio file",
        a nonsense message for a checkpoint policy refusal.
        """
        from flaime_serving.inference import RemoteCheckpointRefused

        assert issubclass(RemoteCheckpointRefused, FileNotFoundError)

    def test_local_checkpoints_are_unaffected_by_the_gate(
        self, tmp_path, mocker
    ) -> None:
        """The gate fires only for remote IDs, never for local checkpoints."""
        from flaime_serving.inference import ASRInferenceEngine

        ckpt = tmp_path / "model"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        mocker.patch(
            "flaime_serving.inference.ASRModelFactory",
            autospec=True,
        ).return_value.create_from_pretrained.return_value = _StubModel()

        engine = ASRInferenceEngine.load(ckpt, model_type="xeus", device="cpu")
        assert isinstance(engine, ASRInferenceEngine)

    def test_load_missing_relative_path_with_multiple_slashes_raises(self) -> None:
        """A relative path with >1 slash (e.g. 'some/nested/path') is NOT a Hub ID
        and must still raise FileNotFoundError when the path doesn't exist locally."""
        from flaime_serving.inference import ASRInferenceEngine

        with pytest.raises(FileNotFoundError):
            ASRInferenceEngine.load("some/nested/path", model_type="wav2vec2")


# ---------------------------------------------------------------------------
# ASRInferenceEngine.transcribe()
# ---------------------------------------------------------------------------


class TestASRInferenceEngineTranscribe:
    def test_transcribe_wav_path_returns_transcription_result(
        self, stub_engine, wav_path
    ) -> None:
        from flaime_serving.inference import TranscriptionResult

        result = stub_engine.transcribe(wav_path, language="en")
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)

    def test_transcribe_numpy_returns_transcription_result(self, stub_engine) -> None:
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        result = stub_engine.transcribe(audio, language="en", sample_rate=TARGET_SR)
        assert isinstance(result.text, str)

    def test_transcribe_tensor_returns_transcription_result(self, stub_engine) -> None:
        audio = torch.zeros(TARGET_SR)
        result = stub_engine.transcribe(audio, language="en", sample_rate=TARGET_SR)
        assert isinstance(result.text, str)

    def test_wav_float32_array_tensor_parity(
        self, stub_engine, sine_wav_path, monkeypatch
    ) -> None:
        """Sine-tone WAV / float32 ndarray / float32 Tensor must reach forward() identically.

        Zero-valued audio cannot expose scaling or channel-axis bugs; a 440 Hz
        sine tone does. The spy captures the input_features tensor so that any
        numeric drift between the three paths surfaces as a test failure, not a
        silent wrong result.
        """
        import torchaudio

        waveform, sr = torchaudio.load(str(sine_wav_path))
        float_arr = waveform.squeeze().numpy()
        float_tensor = waveform.squeeze()

        captured: list[torch.Tensor] = []
        _real_forward = _StubModel.forward

        def _spy(
            self: _StubModel, input_features: torch.Tensor, **kwargs: object
        ) -> dict[str, torch.Tensor]:
            captured.append(input_features.clone())
            return _real_forward(self, input_features, **kwargs)

        monkeypatch.setattr(_StubModel, "forward", _spy)

        stub_engine.transcribe(sine_wav_path, language="en")
        stub_engine.transcribe(float_arr, language="en", sample_rate=sr)
        stub_engine.transcribe(float_tensor, language="en", sample_rate=sr)

        assert len(captured) == 3
        torch.testing.assert_close(captured[0], captured[1])
        torch.testing.assert_close(captured[0], captured[2])

    def test_int16_array_parity_with_float32_array(
        self, stub_engine, sine_wav_path, monkeypatch
    ) -> None:
        """int16 PCM ndarray must reach forward() within int16 quantization error of float32."""
        import torchaudio

        waveform, sr = torchaudio.load(str(sine_wav_path))
        float_arr = waveform.squeeze().numpy()
        # Reconstruct int16 PCM from the float32 reference (≤1 LSB quantization error).
        int16_arr = (float_arr * np.iinfo(np.int16).max).astype(np.int16)

        captured: list[torch.Tensor] = []
        _real_forward = _StubModel.forward

        def _spy(
            self: _StubModel, input_features: torch.Tensor, **kwargs: object
        ) -> dict[str, torch.Tensor]:
            captured.append(input_features.clone())
            return _real_forward(self, input_features, **kwargs)

        monkeypatch.setattr(_StubModel, "forward", _spy)

        stub_engine.transcribe(float_arr, language="en", sample_rate=sr)
        stub_engine.transcribe(int16_arr, language="en", sample_rate=TARGET_SR)

        # Allow 1 LSB of int16 quantization error (1/32767 ≈ 3e-5).
        torch.testing.assert_close(captured[0], captured[1], atol=1e-4, rtol=0.0)

    def test_result_language_matches_input_code(self, stub_engine, wav_path) -> None:
        result = stub_engine.transcribe(wav_path, language="de")
        assert result.language == "de"

    def test_result_latency_ms_is_non_negative(self, stub_engine, wav_path) -> None:
        result = stub_engine.transcribe(wav_path, language="en")
        assert result.latency_ms >= 0.0

    def test_result_model_revision_matches_engine(self, stub_engine, wav_path) -> None:
        result = stub_engine.transcribe(wav_path, language="en")
        assert result.model_revision == _StubModel.revision

    def test_result_decoder_matches_engine(self, stub_engine, wav_path) -> None:
        result = stub_engine.transcribe(wav_path, language="en")
        assert result.decoder == "ctc_greedy"

    def test_transcribe_without_language_defaults_to_none(
        self, stub_engine, wav_path
    ) -> None:
        """Spec: ``language: str | None = None`` — caller may omit it."""
        result = stub_engine.transcribe(wav_path)
        assert result.language is None


class TestTranscriptionResultOptionalFields:
    """Per DEMO-01 spec: language and confidence may be None."""

    def test_language_field_accepts_none(self) -> None:
        from flaime_serving.inference import TranscriptionResult

        r = TranscriptionResult(
            text="hi",
            language=None,
            confidence=0.8,
            latency_ms=1.0,
            model_revision="rev",
            decoder="ctc_greedy",
        )
        assert r.language is None

    def test_confidence_field_accepts_none(self) -> None:
        from flaime_serving.inference import TranscriptionResult

        r = TranscriptionResult(
            text="hi",
            language="en",
            confidence=None,
            latency_ms=1.0,
            model_revision="rev",
            decoder="ctc_greedy",
        )
        assert r.confidence is None

    def test_json_round_trip_with_none_fields(self) -> None:
        from flaime_serving.inference import TranscriptionResult

        original = TranscriptionResult(
            text="hi",
            language=None,
            confidence=None,
            latency_ms=1.0,
            model_revision="rev",
            decoder="ctc_greedy",
        )
        restored = TranscriptionResult.from_json(original.to_json())
        assert restored == original


# ---------------------------------------------------------------------------
# Data sovereignty — no audio persistence
# ---------------------------------------------------------------------------


class TestNoAudioPersistence:
    """Data-sovereignty guarantee: transcribe() must never write any file.

    Tests block write-capable APIs (builtins.open in write mode, torch.save,
    torchaudio.save) so any disk write surfaces as an immediate AssertionError
    regardless of path, extension, or format.
    """

    def _block_all_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch every write-capable API used in or near the inference path."""
        import builtins

        import torchaudio as _ta

        _real_open = builtins.open

        def _no_write_open(
            file: object, mode: str = "r", *args: object, **kwargs: object
        ) -> object:
            if any(c in mode for c in "wax"):
                raise AssertionError(
                    f"transcribe() attempted a file write: open({file!r}, mode={mode!r})"
                )
            return _real_open(file, mode, *args, **kwargs)  # type: ignore[return-value,call-overload]

        def _no_save(*args: object, **kwargs: object) -> None:
            raise AssertionError("Serialisation called during transcribe()")

        monkeypatch.setattr(builtins, "open", _no_write_open)
        monkeypatch.setattr(torch, "save", _no_save)
        monkeypatch.setattr(_ta, "save", _no_save)

    def test_transcribe_wav_path_writes_no_files(
        self, stub_engine, wav_path, monkeypatch
    ) -> None:
        self._block_all_writes(monkeypatch)
        stub_engine.transcribe(wav_path, language="en")

    def test_transcribe_numpy_input_writes_no_files(
        self, stub_engine, monkeypatch
    ) -> None:
        self._block_all_writes(monkeypatch)
        audio = np.zeros(TARGET_SR, dtype=np.float32)
        stub_engine.transcribe(audio, language="en", sample_rate=TARGET_SR)
