"""Test suite for flaime_serving.cli (DEMO-01), moved from FLAIME's
tests/serving/test_inference.py (TestCLIArgparse, TestCLIRunTranscribe)
as part of 26Q3-REPO-06, alongside the serve.py -> cli.py move.

Run: uv run pytest tests/test_cli.py -v
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def stub_result():
    """Pre-built TranscriptionResult for CLI output tests."""
    from flaime_serving.inference import TranscriptionResult

    return TranscriptionResult(
        text="hello world",
        language="en",
        confidence=0.95,
        latency_ms=42.0,
        model_revision="stub-rev-abc123",
        decoder="ctc_greedy",
    )


# ---------------------------------------------------------------------------
# CLI — argparse structure
# ---------------------------------------------------------------------------


class TestCLIArgparse:
    def test_transcribe_defaults_to_refusing_remote_checkpoints(self) -> None:
        """Offline by default at the CLI too, not only in the demo."""
        args = self._parse(
            ["transcribe", "audio.wav", "--checkpoint", "/ckpt", "--model-type", "xeus"]
        )
        assert args.allow_remote is False

    def test_transcribe_accepts_an_explicit_remote_opt_in(self) -> None:
        """`--allow-remote` keeps Hub IDs usable, e.g. facebook/wav2vec2-base-960h."""
        args = self._parse(
            [
                "transcribe",
                "audio.wav",
                "--checkpoint",
                "facebook/wav2vec2-base-960h",
                "--model-type",
                "wav2vec2",
                "--allow-remote",
            ]
        )
        assert args.allow_remote is True

    def _parse(self, argv: list[str]) -> argparse.Namespace:
        from flaime_serving.cli import _build_parser

        return _build_parser().parse_args(argv)

    def test_serve_transcribe_positional_and_required_flags(self) -> None:
        args = self._parse(
            [
                "transcribe",
                "audio.wav",
                "--checkpoint",
                "/ckpt",
                "--model-type",
                "xeus",
            ]
        )
        assert args.audio_path == "audio.wav"
        assert args.checkpoint == "/ckpt"
        assert args.model_type == "xeus"

    def test_serve_transcribe_defaults(self) -> None:
        args = self._parse(
            [
                "transcribe",
                "x.wav",
                "--checkpoint",
                "/c",
                "--model-type",
                "xeus",
            ]
        )
        assert args.device == "cpu"
        assert args.decoder == "ctc_greedy"
        assert args.language == "und"
        assert args.json is False

    def test_json_flag(self) -> None:
        args = self._parse(
            [
                "transcribe",
                "x.wav",
                "--checkpoint",
                "/c",
                "--model-type",
                "xeus",
                "--json",
            ]
        )
        assert args.json is True

    def test_short_flags_accepted(self) -> None:
        args = self._parse(
            ["transcribe", "x.wav", "-c", "/c", "-m", "wav2vec2", "-l", "de"]
        )
        assert args.checkpoint == "/c"
        assert args.model_type == "wav2vec2"
        assert args.language == "de"


# ---------------------------------------------------------------------------
# CLI — run_transcribe output
# ---------------------------------------------------------------------------


class TestCLIRunTranscribe:
    def test_json_mode_prints_parseable_json(self, stub_result, capsys, mocker) -> None:
        from flaime_serving import cli

        mocker.patch(
            "flaime_serving.cli.ASRInferenceEngine.load",
            return_value=MagicMock(transcribe=MagicMock(return_value=stub_result)),
        )
        args = argparse.Namespace(
            audio_path="x.wav",
            checkpoint="/ckpt",
            model_type="xeus",
            language="en",
            device="cpu",
            decoder="ctc_greedy",
            json=True,
        )
        exit_code = cli.run_transcribe(args)
        assert exit_code == 0
        out, _ = capsys.readouterr()
        parsed = json.loads(out)
        assert parsed["text"] == stub_result.text

    def test_text_mode_prints_plain_text(self, stub_result, capsys, mocker) -> None:
        from flaime_serving import cli

        mocker.patch(
            "flaime_serving.cli.ASRInferenceEngine.load",
            return_value=MagicMock(transcribe=MagicMock(return_value=stub_result)),
        )
        args = argparse.Namespace(
            audio_path="x.wav",
            checkpoint="/ckpt",
            model_type="xeus",
            language="en",
            device="cpu",
            decoder="ctc_greedy",
            json=False,
        )
        exit_code = cli.run_transcribe(args)
        assert exit_code == 0
        out, _ = capsys.readouterr()
        assert stub_result.text in out
        # Must not be a JSON object — plain text only
        with pytest.raises(json.JSONDecodeError):
            json.loads(out)
