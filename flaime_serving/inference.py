"""ASR inference engine for the FLAIME demo serving pipeline (DEMO-01)."""

from __future__ import annotations

import dataclasses
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from flaime_serving.ctc_beam_search import beam_search_ctc_decode
from flaime_serving.vendored.model_factory import ASRModelFactory

TARGET_SR = 16_000  # Hz — all audio inputs are normalised to this rate


@dataclasses.dataclass
class TranscriptionResult:
    """Typed result returned by ASRInferenceEngine.transcribe().

    Attributes:
        text: Decoded transcript.
        language: BCP-47/ISO 639-3 code passed by the caller, or ``None`` when
            the caller did not specify one (per DEMO-01 spec).
        confidence: Frame-level acoustic confidence heuristic in (0, 1], or
            ``None`` if the decoder did not produce a usable score.  Computed
            as exp(mean(max log-prob per frame)) over all encoder output
            frames.  This is NOT the probability of the decoded sequence —
            CTC sequence probability requires marginalising over all
            alignments and is not computed here.  Treat this value as a rough
            signal of acoustic certainty, not a calibrated probability.
        latency_ms: Wall-clock time from audio input to result, in ms.
        model_revision: Checkpoint revision tag.
        decoder: Decoding strategy used (e.g. "ctc_greedy", "ctc_beam5").
    """

    text: str
    language: str | None
    confidence: float | None
    latency_ms: float
    model_revision: str
    decoder: str

    def to_json(self) -> str:
        """Serialise to a JSON string.

        Returns:
            JSON representation of all fields.
        """
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> TranscriptionResult:
        """Deserialise from a JSON string produced by to_json().

        Args:
            s: JSON string.

        Returns:
            TranscriptionResult instance.
        """
        return cls(**json.loads(s))


def _normalize_audio(
    audio: str | Path | np.ndarray | torch.Tensor,
    sample_rate: int | None = None,
) -> np.ndarray:
    """Convert any supported audio input to a 16 kHz mono float32 numpy array.

    Args:
        audio: One of:
            - str or Path: path to a WAV file (sample_rate read from file header)
            - np.ndarray: waveform, shape (T,) or (C, T); float or int.
              Integer dtypes (e.g. int16 PCM) are scaled to [-1, 1] by
              dividing by iinfo(dtype).max before conversion to float32.
            - torch.Tensor: waveform, shape (T,) or (C, T); float or int.
              Same integer-scaling rule applies.
            Stereo inputs are mean-collapsed to mono before resampling.
            For ndarray/Tensor, the channel axis is inferred as whichever
            dimension is smaller, so both (C, T) from torchaudio and
            (T, C) from soundfile are handled correctly.
            WAV file paths always return float32 in [-1, 1] (torchaudio
            default), so all three input types produce equivalent outputs
            for the same audio content.
        sample_rate: Original sample rate in Hz.  Required for ndarray/Tensor
            inputs; ignored (overwritten) for file path inputs.

    Returns:
        1-D float32 numpy array at TARGET_SR (16 000 Hz).

    Raises:
        FileNotFoundError: If audio is a path that does not exist.
        ValueError: If sample_rate is None for ndarray/Tensor inputs.
    """
    if isinstance(audio, str | Path):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate required for ndarray audio input.")
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        t = torch.from_numpy(np.array(audio, dtype=np.float32))
        if t.ndim == 2:
            t = t.mean(dim=0 if t.shape[0] <= t.shape[1] else 1)
        waveform, sr = t, sample_rate
    elif isinstance(audio, torch.Tensor):
        if sample_rate is None:
            raise ValueError("sample_rate required for Tensor audio input.")
        if not audio.is_floating_point():
            t = audio.to(torch.float32) / torch.iinfo(audio.dtype).max
        else:
            t = audio.float()
        if t.ndim == 2:
            t = t.mean(dim=0 if t.shape[0] <= t.shape[1] else 1)
        waveform, sr = t, sample_rate
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if sr <= 0:
        raise ValueError(
            f"Invalid sample_rate={sr}. Sample rate must be a positive integer."
        )

    if sr != TARGET_SR:
        warnings.warn(
            f"Resampling audio from {sr} Hz to {TARGET_SR} Hz.",
            UserWarning,
            stacklevel=2,
        )
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=TARGET_SR
        )

    return waveform.detach().cpu().numpy().astype(np.float32)


def _validate_decoder(decoder: str) -> None:
    """Raise ValueError if decoder is not a recognised strategy string."""
    if decoder != "ctc_greedy" and not re.fullmatch(r"ctc_beam\d+", decoder):
        raise ValueError(
            f"Unrecognised decoder: {decoder!r}. "
            "Expected 'ctc_greedy' or 'ctc_beam<N>'."
        )


class ASRInferenceEngine:
    """Audio-in → text-out engine for the FLAIME community demo.

    Thin wrapper around a loaded ASR model and a CTC decoder.  The public
    interface is intentionally small: load() and transcribe().  Audio bytes
    are never written to disk (Indigenous data-sovereignty constraint).

    Usage::

        engine = ASRInferenceEngine.load(
            checkpoint_path="/checkpoints/xeus-en-v1",
            model_type="xeus",
            device="cpu",
            decoder="ctc_greedy",
        )
        result = engine.transcribe("utterance.wav", language="en")
        print(result.text)
    """

    def __init__(
        self,
        *,
        model: Any,
        model_revision: str,
        decoder: str,
        device: str,
    ) -> None:
        """Direct constructor — prefer ASRInferenceEngine.load() for production use.

        Args:
            model: Loaded ASR model object satisfying ASRModelProtocol.  Must
                expose forward(input_features, wav_lengths=...) returning a
                dict with a "logits" key, and a processor property whose
                decode(token_ids) converts integer token IDs to text.
            model_revision: Revision string from the checkpoint (e.g., a git SHA
                or HuggingFace model revision tag).
            decoder: Decoding strategy — "ctc_greedy" or "ctc_beam<N>" where N
                is the integer beam width (e.g., "ctc_beam5").
            device: Torch device string ("cpu", "cuda", "cuda:0", …).

        Raises:
            ValueError: If decoder is not "ctc_greedy" or "ctc_beam<N>".
        """
        _validate_decoder(decoder)
        self.model = model
        self.model_revision = model_revision
        self.decoder = decoder
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        model_type: str,
        device: str | None = None,
        decoder: str = "ctc_greedy",
        warmup: bool = False,
    ) -> ASRInferenceEngine:
        """Load a checkpoint and return a ready-to-use engine.

        Args:
            checkpoint_path: Directory containing model weights and config.json,
                as produced by the FLAIME training pipeline.
            model_type: Architecture key registered in ASRModelFactory
                (e.g., "xeus", "whisper", "wav2vec2").
            device: Torch device string. ``None`` (default) auto-detects:
                ``"cuda"`` when a CUDA device is available, ``"cpu"`` otherwise.
                Per DEMO-01 acceptance criteria.
            decoder: Decoding strategy — "ctc_greedy" or "ctc_beam<N>".
            warmup: When ``True``, run one throwaway synthetic inference before
                returning so the first user-facing request doesn't pay
                cold-start cost.  Uses in-RAM silence only — no disk/network
                I/O.  Defaults to ``False`` to keep ``load()`` side-effect-free.

        Returns:
            Initialised ASRInferenceEngine with model already on device.

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
            ValueError: If model_type is not registered, or decoder is malformed.
        """
        checkpoint_path = str(checkpoint_path)
        path = Path(checkpoint_path)
        # A HuggingFace Hub ID (e.g. "facebook/wav2vec2-base-960h") is not a
        # local path: it is relative, does not yet exist on disk, and contains
        # exactly one "/".  Skip the existence check for those; the model's
        # own from_pretrained will raise a clear error if the ID is invalid.
        is_hf_id = (
            not path.is_absolute()
            and not path.exists()
            and checkpoint_path.count("/") == 1
        )
        if not is_hf_id and not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        _validate_decoder(decoder)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        factory = ASRModelFactory()
        model = factory.create_from_pretrained(model_type, checkpoint_path)
        model.to(device)  # type: ignore[arg-type]
        model.eval()

        # cfg_file only exists for local FLAIME-trained checkpoints
        cfg_file = path / "config.json"
        model_revision = path.name
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text())
            model_revision = cfg.get("model_revision", path.name)

        engine = cls(
            model=model, model_revision=model_revision, decoder=decoder, device=device
        )
        if warmup:
            engine.warmup()
        return engine

    def transcribe(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        language: str | None = None,
        *,
        sample_rate: int | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single utterance.

        Audio bytes are processed entirely in RAM and never written to disk
        (Indigenous data-sovereignty constraint — non-negotiable).

        Args:
            audio: Input audio as a WAV file path, 1-D numpy waveform, or
                1-D torch waveform.  Stereo inputs are collapsed to mono.
            language: BCP-47 or ISO 639-3 language code (e.g., "en", "de"),
                or ``None`` (default) when the caller has no language tag to
                attach.  Stored verbatim in the result; may gate future
                language-specific post-processing.
            sample_rate: Original sample rate in Hz.  Required when audio is an
                ndarray or Tensor; ignored for file path inputs.

        Returns:
            TranscriptionResult with all fields populated.

        Raises:
            ValueError: If the normalised waveform is empty.
        """
        start = time.perf_counter()

        waveform = _normalize_audio(audio, sample_rate)
        if len(waveform) == 0:
            raise ValueError("Empty waveform after normalisation.")

        input_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        wav_lengths = torch.tensor(
            [input_tensor.shape[1]], dtype=torch.long, device=self.device
        )

        with torch.inference_mode():
            outputs = self.model.forward(input_tensor, wav_lengths=wav_lengths)

        logits = outputs["logits"]
        log_probs = F.log_softmax(logits, dim=-1)

        beam_width = 1 if self.decoder == "ctc_greedy" else int(self.decoder[8:])
        token_ids = beam_search_ctc_decode(log_probs, beam_width)[0]

        text = self.model.processor.decode(token_ids)
        # Heuristic: geometric mean of the per-frame peak probability.
        # Not a sequence probability — see TranscriptionResult.confidence docstring.
        confidence = float(log_probs[0].max(dim=-1).values.mean().exp())
        latency_ms = (time.perf_counter() - start) * 1000

        return TranscriptionResult(
            text=text,
            language=language,
            confidence=confidence,
            latency_ms=latency_ms,
            model_revision=self.model_revision,
            decoder=self.decoder,
        )

    def warmup(self, duration_s: float = 1.0) -> None:
        """Run one throwaway inference so the first user request isn't cold.

        The first forward pass through a freshly-loaded model pays lazy
        initialisation (cuDNN autotune, kernel JIT, allocator warmup) that a
        presenter would otherwise feel as a one-off stall on the demo's first
        request.  Calling ``warmup()`` at ``load()`` time moves that cost off
        the user-facing path.

        Synthetic silence is generated in RAM at the target sample rate — no
        disk read and no network I/O, so this is safe to call eagerly and
        respects the data-sovereignty "audio never touches disk" constraint.

        Args:
            duration_s: Length of the synthetic warmup clip in seconds.  One
                second is enough to exercise the full forward + decode path
                while keeping startup cost negligible.

        Returns:
            None.  The throwaway transcript is discarded.
        """
        samples = max(1, int(duration_s * TARGET_SR))
        synthetic = np.zeros(samples, dtype=np.float32)
        self.transcribe(synthetic, sample_rate=TARGET_SR)
