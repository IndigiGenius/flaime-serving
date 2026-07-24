"""Base ASR model abstraction.

This module provides the base classes for all ASR model architectures in FLAIME.
It defines a common interface that all models (Whisper, Wav2Vec2, MMS, etc.)
must implement, enabling model-agnostic training and evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from flaime_serving.vendored.packing_config import PackingConfig


@dataclass
class BaseASRModelConfig:
    """Configuration for ASR models.

    This dataclass holds configuration parameters common to all ASR model types.
    Each model type can extend this with model-specific parameters.

    Attributes:
        model_type: Type of model (e.g., "whisper", "wav2vec2", "mms", "conformer")
        model_name_or_path: HuggingFace model identifier or local path
        loss_type: Type of loss function ("cross-entropy", "ctc", "rnnt")
        device: Device to run the model on ("cuda" or "cpu")

    Examples:
        >>> config = BaseASRModelConfig(
        ...     model_type="whisper",
        ...     model_name_or_path="openai/whisper-tiny",
        ...     loss_type="cross-entropy",
        ...     device="cuda"
        ... )
    """

    model_type: str
    model_name_or_path: str
    loss_type: str = "ctc"
    device: str | None = None

    def __post_init__(self):
        """Initialize device based on CUDA availability if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@runtime_checkable
class ASRModelProtocol(Protocol):
    """Protocol for ASR models consumed by the Trainer.

    This protocol decouples the Trainer from the concrete BaseASRModel hierarchy,
    enabling static type checking without requiring inheritance.

    Any nn.Module that implements forward(), generate(), save_pretrained(), and
    exposes a processor property satisfies this protocol.
    """

    training: bool

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict: ...

    def generate(self, input_features: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...

    def save_pretrained(self, save_path: str) -> None: ...

    @property
    def processor(self) -> Any: ...

    def parameters(self, recurse: bool = True) -> Any: ...

    def train(self, mode: bool = True) -> "ASRModelProtocol": ...

    def eval(self) -> "ASRModelProtocol": ...

    def to(self, *args: Any, **kwargs: Any) -> "ASRModelProtocol": ...


class BaseASRModel(nn.Module, ABC):
    """Abstract base class for all ASR models.

    This class defines the common interface that all ASR model implementations
    must provide. It inherits from both nn.Module (for PyTorch integration)
    and ABC (for abstract method enforcement).

    All subclasses must implement:
        - forward(): Forward pass with loss computation
        - generate(): Inference/generation method
        - save_pretrained(): Save model checkpoint
        - from_pretrained(): Load model from checkpoint
        - processor: Property to access model's processor/tokenizer

    Examples:
        >>> class MyASRModel(BaseASRModel):
        ...     def forward(self, input_features, labels=None):
        ...         # Implementation
        ...         pass
        ...
        ...     def generate(self, input_features, **kwargs):
        ...         # Implementation
        ...         pass
        ...
        ...     # ... implement other abstract methods
    """

    @abstractmethod
    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict:
        """Forward pass through the model.

        Args:
            input_features: Input audio features, shape (batch, features, time)
            labels: Optional target labels for training, shape (batch, seq_len)

        Returns:
            Dictionary containing at least:
                - "logits": Model predictions, shape (batch, seq_len, vocab_size)
                - "loss": Training loss (only when labels are provided)
        """
        pass

    @abstractmethod
    def generate(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate transcriptions from audio features.

        This method is used for inference/evaluation. It should perform
        decoding/generation to produce final transcriptions.

        Args:
            input_features: Input audio features, shape (batch, features, time)
            **kwargs: Additional generation parameters (e.g., max_length, num_beams)

        Returns:
            Generated token IDs, shape (batch, generated_length)

        Examples:
            >>> generated = model.generate(
            ...     input_features,
            ...     max_length=100,
            ...     num_beams=5
            ... )
        """
        pass

    @abstractmethod
    def save_pretrained(self, save_path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            save_path: Directory path to save the model

        Examples:
            >>> model.save_pretrained("/path/to/checkpoint")
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, load_path: str, config: BaseASRModelConfig | None = None
    ) -> "BaseASRModel":
        """Load model from a saved checkpoint.

        Args:
            load_path: Directory path to load the model from
            config: Optional model configuration. If None, will be inferred.

        Returns:
            Loaded model instance

        Examples:
            >>> model = MyASRModel.from_pretrained("/path/to/checkpoint")
        """
        pass

    @property
    @abstractmethod
    def processor(self):
        """Get the model's processor/tokenizer.

        Returns:
            Model processor for encoding/decoding text and audio

        Examples:
            >>> processor = model.processor
            >>> text = processor.decode(token_ids)
        """
        pass

    @abstractmethod
    def get_collate_fn(
        self,
        languages: dict[str, str],
        packing_config: PackingConfig | None = None,
    ) -> Any:
        """Get the appropriate collate function for multilingual training.

        Args:
            languages: Dictionary of language codes to names.
            packing_config: Optional packing configuration for sample packing.

        Returns:
            Collate function for DataLoader.
        """
        pass

    @abstractmethod
    def get_expert_collate_fn(
        self,
        lang_code: str,
        packing_config: PackingConfig | None = None,
    ) -> Any:
        """Get the appropriate collate function for single-language expert training.

        Args:
            lang_code: Language code for expert training.
            packing_config: Optional packing configuration for sample packing.

        Returns:
            Collate function for DataLoader.
        """
        pass
