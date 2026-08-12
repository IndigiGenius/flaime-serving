"""ASR Model Protocol definitions for model-agnostic training.

This module defines runtime-checkable protocols that establish a common interface
for ASR models in FLAIME.
"""

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class ASRModelProtocol(Protocol):
    """Protocol defining the core interface for ASR models."""

    @property
    def processor(self) -> Any:
        """Get the model's processor for audio/text processing."""
        ...

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through the model."""
        ...

    def generate(
        self,
        input_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate transcriptions from audio features."""
        ...


@runtime_checkable
class CTCCapable(Protocol):
    """Protocol for models that support CTC output.

    CTC-capable models have a ctc_projection attribute that maps encoder
    hidden states to vocabulary logits for CTC loss computation.
    """

    ctc_projection: nn.Module | None
    """CTC projection layer mapping encoder hidden states to vocab logits."""


@runtime_checkable
class RNNTCapable(Protocol):
    """Protocol for models with RNN-T decoder.

    RNN-T models have a rnnt_decoder attribute containing prediction
    and joint networks for transducer-based training.
    """

    rnnt_decoder: nn.Module | None
    """RNN-T decoder module with prediction and joint networks."""


@runtime_checkable
class IntermediateOutputCapable(Protocol):
    """Protocol for models that output intermediate layer representations.

    Models with intermediate decoders can provide auxiliary supervision
    from encoder layers during training.
    """

    intermediate_decoders: nn.Module | None
    """Module that captures and projects intermediate layer outputs."""


def supports_ctc(model: Any) -> bool:
    """Check if a model supports CTC output.

    A model supports CTC if it has a ctc_projection attribute that is not None.
    Uses hasattr for reliable instance attribute checking (isinstance with
    Protocol only works for class-level attributes).

    Args:
        model: Model instance to check

    Returns:
        True if model has CTC capability with a valid projection layer
    """
    return hasattr(model, "ctc_projection") and model.ctc_projection is not None


def supports_rnnt(model: Any) -> bool:
    """Check if a model supports RNN-T output.

    Args:
        model: Model instance to check

    Returns:
        True if model has RNN-T capability with a valid decoder
    """
    return hasattr(model, "rnnt_decoder") and model.rnnt_decoder is not None


def supports_intermediate_outputs(model: Any) -> bool:
    """Check if a model supports intermediate layer outputs.

    Args:
        model: Model instance to check

    Returns:
        True if model has intermediate output capability
    """
    return (
        hasattr(model, "intermediate_decoders")
        and model.intermediate_decoders is not None
    )


def get_model_capabilities(model: Any) -> dict[str, bool]:
    """Get a summary of all capabilities for a model.

    Useful for logging and debugging to understand what loss types
    a model can support.

    Args:
        model: Model instance to check

    Returns:
        Dictionary mapping capability names to boolean values
    """
    return {
        "asr_protocol": isinstance(model, ASRModelProtocol),
        "ctc": supports_ctc(model),
        "rnnt": supports_rnnt(model),
        "intermediate_outputs": supports_intermediate_outputs(model),
    }
