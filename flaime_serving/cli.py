"""``flaime-serve`` — transcription CLI and demo UI launcher (DEMO-01, DEMO-04).

Sub-command tree::

    flaime-serve transcribe <audio_path>
        --checkpoint <path>      (required)
        --model-type <name>      (required)
        [--language <code>]      (default: "und")
        [--device <str>]         (default: "cpu")
        [--decoder <str>]        (default: "ctc_greedy")
        [--json]                 emit TranscriptionResult as JSON

    flaime-serve ui
        [--checkpoint <path>]
        [--languages-config <path>]
        [--model-type <name>]    (default: "xeus")
        [--device <str>]         (default: auto)
        [--decoder <str>]        (default: "ctc_greedy")
        [--bind <addr>]          (default: "127.0.0.1")
        [--port <int>]           (default: 8501)

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from flaime_serving.inference import ASRInferenceEngine

# Absolute path to the Streamlit app so `flaime-serve ui` works regardless of cwd.
_APP_PY = Path(__file__).parent.parent / "apps" / "demo" / "app.py"


def _add_transcribe_parser(
    serve_subs: argparse._SubParsersAction[Any],
) -> argparse.ArgumentParser:
    """Register ``transcribe`` and return its parser.

    Args:
        serve_subs: Subparsers action on the top-level flaime-serve parser.

    Returns:
        The transcribe ArgumentParser.
    """
    p = serve_subs.add_parser(
        "transcribe",
        help="Transcribe a WAV file using a FLAIME checkpoint.",
    )
    p.add_argument("audio_path", help="Path to a 16 kHz mono WAV file.")
    p.add_argument(
        "--checkpoint", "-c", required=True, help="Checkpoint directory or file."
    )
    p.add_argument(
        "--model-type",
        "-m",
        required=True,
        dest="model_type",
        help="Model architecture key (e.g., xeus, whisper, wav2vec2).",
    )
    p.add_argument(
        "--language",
        "-l",
        default="und",
        help="BCP-47/ISO 639-3 language code (default: und).",
    )
    p.add_argument(
        "--device", default="cpu", help="Torch device string (default: cpu)."
    )
    p.add_argument(
        "--decoder",
        default="ctc_greedy",
        help="Decoding strategy: ctc_greedy or ctc_beam<N> (default: ctc_greedy).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit TranscriptionResult as JSON instead of plain text.",
    )
    p.set_defaults(func=run_transcribe)
    return p


def run_transcribe(args: argparse.Namespace) -> int:
    """Execute ``flaime-serve transcribe``.

    Args:
        args: Parsed namespace; expected attributes: audio_path, checkpoint,
            model_type, language, device, decoder, json (bool).

    Returns:
        Exit code — 0 on success, non-zero on failure.
    """
    # catch exceptions during transcription to prevent crashes and provide feedback
    try:
        engine = ASRInferenceEngine.load(
            args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            decoder=args.decoder,
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}", file=sys.stderr)
        return 2

    try:
        result = engine.transcribe(args.audio_path, language=args.language)
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(result.to_json())
    else:
        print(result.text)
    return 0


def _add_ui_parser(
    serve_subs: argparse._SubParsersAction[Any],
) -> argparse.ArgumentParser:
    """Register ``ui`` and return its parser.

    Args:
        serve_subs: Subparsers action on the top-level flaime-serve parser.

    Returns:
        The ui ArgumentParser.
    """
    p = serve_subs.add_parser(
        "ui",
        help="Launch the Streamlit demo UI (DEMO-04).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a FLAIME checkpoint directory. "
        "If omitted the UI starts without a loaded model.",
    )
    p.add_argument(
        "--languages-config",
        dest="languages_config",
        default=None,
        metavar="PATH",
        help="Path to the YAML language-routing config "
        "(e.g. configs/serving/demo_languages.yaml). "
        "When set, --checkpoint is ignored and the router selects per language.",
    )
    p.add_argument(
        "--model-type",
        dest="model_type",
        default="xeus",
        metavar="TYPE",
        help="Model architecture key (e.g. xeus, whisper). Default: xeus.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Torch device string (e.g. cpu, cuda). Auto-detects if omitted.",
    )
    p.add_argument(
        "--decoder",
        default="ctc_greedy",
        metavar="DECODER",
        help="Decoding strategy: ctc_greedy or ctc_beam<N>. Default: ctc_greedy.",
    )
    p.add_argument(
        "--bind",
        default="127.0.0.1",
        metavar="ADDR",
        help="Address to bind (default: 127.0.0.1). "
        "Use 0.0.0.0 to expose on the local network (shows a warning banner).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8501,
        metavar="PORT",
        help="Port to listen on (default: 8501).",
    )
    p.set_defaults(func=run_ui)
    return p


def run_ui(args: argparse.Namespace) -> int:
    """Execute ``flaime-serve ui`` — launch the Streamlit demo UI.

    Sets the env vars that apps/demo/app.py reads at Streamlit runtime, then
    spawns Streamlit directly (no extra subprocess layer through app.py).

    Args:
        args: Parsed namespace from _add_ui_parser().

    Returns:
        Exit code from the Streamlit subprocess.
    """
    if args.languages_config:
        os.environ["DEMO_LANGUAGES_CONFIG"] = args.languages_config
        os.environ.pop("DEMO_CHECKPOINT", None)
        os.environ.pop("DEMO_MODEL_TYPE", None)
    elif args.checkpoint:
        os.environ["DEMO_CHECKPOINT"] = args.checkpoint
        os.environ["DEMO_MODEL_TYPE"] = args.model_type

    if args.device:
        os.environ["DEMO_DEVICE"] = args.device
    os.environ["DEMO_DECODER"] = args.decoder

    if args.bind != "127.0.0.1":
        os.environ["DEMO_PUBLIC_BIND"] = "1"

    if not _APP_PY.exists():
        print(
            f"Error: demo app not found at {_APP_PY}. Run from the FLAIME repo root.",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP_PY),
        "--server.address",
        args.bind,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    result = subprocess.run(cmd)
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``flaime-serve`` parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="flaime-serve",
        description="Serve a trained checkpoint for single-utterance transcription.",
    )
    subs = parser.add_subparsers(
        dest="serve_subcommand",
        metavar="{transcribe,ui}",
        required=True,
    )
    _add_transcribe_parser(subs)
    _add_ui_parser(subs)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point for the ``flaime-serve`` console script."""
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
