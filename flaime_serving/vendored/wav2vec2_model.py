"""Wav2Vec2 ASR model implementation using BaseASRModel interface.

This module provides a Wav2Vec2ASRModel class that implements the BaseASRModel
interface for Facebook's Wav2Vec2 architecture.
"""

import json
import os
import tempfile
from typing import Any

import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)

from flaime_serving.vendored.packing_config import PackingConfig

from .base_model import BaseASRModel, BaseASRModelConfig


def _is_main_process() -> bool:
    """Check if this is rank 0 (or non-distributed)."""
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _create_placeholder_tokenizer() -> Wav2Vec2CTCTokenizer:
    """Create a minimal CTC tokenizer with only special tokens.

    Used as a placeholder for SSL-only models (e.g. XLS-R) that ship
    without a tokenizer.  ``maybe_expand_ctc_vocab()`` replaces it with
    the real dataset-derived tokenizer before training starts.
    """
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(vocab, f)
        vocab_file = f.name
    try:
        return Wav2Vec2CTCTokenizer(
            vocab_file,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_lower_case=True,
        )
    finally:
        os.unlink(vocab_file)


class Wav2Vec2ASRModel(BaseASRModel):
    """Wav2Vec2 model wrapper for CTC-based ASR training.

    This class implements the BaseASRModel interface for Wav2Vec2 models.
    Wav2Vec2 uses CTC loss by default.

    Attributes:
        config: Model configuration
        model: Underlying HuggingFace Wav2Vec2ForCTC model
        _processor: Wav2Vec2 processor for audio/text processing

    Examples:
        >>> config = BaseASRModelConfig(
        ...     model_type="wav2vec2",
        ...     model_name_or_path="facebook/wav2vec2-base",
        ...     loss_type="ctc",
        ... )
        >>> model = Wav2Vec2ASRModel(config)
    """

    def __init__(
        self,
        config: BaseASRModelConfig,
        from_scratch: bool = False,
        blank_bias_init: float | None = None,
    ) -> None:
        """Initialize Wav2Vec2 model with configuration.

        Args:
            config: Model configuration object
            from_scratch: If True, initialize with random weights instead of pre-trained
            blank_bias_init: If set, initialize lm_head.bias[0] (blank token) to this
                value. Negative values suppress blank from step 0, preventing CTC blank
                collapse. Recommended: -3.0.

        Raises:
            ValueError: If model_type is not "wav2vec2"
        """
        super().__init__()

        if config.model_type.lower() != "wav2vec2":
            raise ValueError(
                f"Wav2Vec2ASRModel requires model_type='wav2vec2', got '{config.model_type}'"
            )

        self.config = config

        # Load processor. SSL-only models (e.g., XLS-R) lack a tokenizer, so
        # fall back to feature_extractor only — maybe_expand_ctc_vocab() will
        # add a dataset-derived tokenizer before training starts.
        try:
            self._processor = Wav2Vec2Processor.from_pretrained(
                config.model_name_or_path
            )
        except (OSError, TypeError):
            # SSL-only models (XLS-R) lack a tokenizer/vocab file —
            # from_pretrained raises TypeError (vocab_file is None) or
            # OSError (file not found). Create a minimal placeholder
            # tokenizer so Wav2Vec2Processor can be constructed;
            # maybe_expand_ctc_vocab() replaces it with the real one
            # derived from training data before training starts.
            if _is_main_process():
                print(
                    f"No tokenizer for {config.model_name_or_path} "
                    "(SSL-only model) — using placeholder tokenizer"
                )
            fe = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name_or_path)
            placeholder_tokenizer = _create_placeholder_tokenizer()
            self._processor = Wav2Vec2Processor(
                feature_extractor=fe, tokenizer=placeholder_tokenizer
            )

        # Load model - either from pretrained or with random initialization
        wav2vec2_config = Wav2Vec2Config.from_pretrained(config.model_name_or_path)
        is_ssl_only = "Wav2Vec2ForCTC" not in (wav2vec2_config.architectures or [])

        if from_scratch:
            # Load config and create model with random weights
            self.model = Wav2Vec2ForCTC(wav2vec2_config)
            if _is_main_process():
                print("Initialized Wav2Vec2 model with RANDOM weights")
        elif is_ssl_only:
            # SSL-only models (e.g. XLS-R) have no CTC head — load the
            # backbone weights into a Wav2Vec2ForCTC wrapper so we get
            # pretrained encoder + randomly initialized lm_head.
            ssl_model = Wav2Vec2Model.from_pretrained(config.model_name_or_path)
            self.model = Wav2Vec2ForCTC(wav2vec2_config)
            self.model.wav2vec2.load_state_dict(ssl_model.state_dict())
            del ssl_model
            if _is_main_process():
                print(
                    f"Loaded SSL backbone from {config.model_name_or_path} "
                    "into Wav2Vec2ForCTC (random lm_head)"
                )
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(config.model_name_or_path)
            if _is_main_process():
                print("Loaded Wav2Vec2 model with PRE-TRAINED weights")

        # Apply blank bias initialization to suppress CTC blank collapse
        if blank_bias_init is not None:
            with torch.no_grad():
                self.model.lm_head.bias.data[0] = blank_bias_init
            if _is_main_process():
                print(f"Set blank bias init: lm_head.bias[0] = {blank_bias_init}")

        # Move model to device
        self.to(config.device)

    def forward(  # type: ignore[override]
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_features: Input audio features, shape (batch, time)
            labels: Optional target labels for training, shape (batch, seq_len)
            **kwargs: Additional args (attention_mask)

        Returns:
            Dict with "logits", "ctc_logits", and optionally "loss"
        """
        attention_mask = kwargs.pop("attention_mask", None)

        # Wav2Vec2ForCTC already handles CTC loss computation internally
        fwd_kwargs: dict[str, Any] = {"input_values": input_features}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if labels is not None:
            fwd_kwargs["labels"] = labels

        model_output = self.model(**fwd_kwargs)

        result: dict[str, torch.Tensor] = {
            "logits": model_output.logits,
            "ctc_logits": model_output.logits,
        }

        if hasattr(model_output, "loss") and model_output.loss is not None:
            result["loss"] = model_output.loss

        return result

    def generate(self, input_features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate transcriptions from input features.

        For CTC models, this performs greedy decoding (argmax).

        Args:
            input_features: Input audio features, shape (batch, time)
            **kwargs: Additional generation arguments (currently unused for CTC)

        Returns:
            Generated token IDs, shape (batch, generated_length)

        Examples:
            >>> generated = model.generate(input_features)
        """
        # Get logits
        with torch.no_grad():
            logits = self.model(input_values=input_features).logits

        # Greedy decoding: take argmax
        predicted_ids = torch.argmax(logits, dim=-1)

        return predicted_ids

    def save_pretrained(self, save_path: str) -> None:
        """Save model weights and configuration.

        Args:
            save_path: Path to save the model

        Examples:
            >>> model.save_pretrained("/path/to/checkpoint")
        """
        self.model.save_pretrained(save_path)
        self._processor.save_pretrained(save_path)

    @classmethod
    def from_pretrained(
        cls, load_path: str, config: BaseASRModelConfig | None = None
    ) -> "Wav2Vec2ASRModel":
        """Load a pretrained model.

        Args:
            load_path: Path to load the model from
            config: Optional configuration (will be inferred if not provided)

        Returns:
            Loaded model instance

        Examples:
            >>> model = Wav2Vec2ASRModel.from_pretrained("/path/to/checkpoint")
        """
        if config is None:
            # Infer configuration from saved model
            config = BaseASRModelConfig(
                model_type="wav2vec2",
                model_name_or_path=load_path,
                loss_type="ctc",
            )
        else:
            # Update config to point to load_path
            config = BaseASRModelConfig(
                model_type="wav2vec2",
                model_name_or_path=load_path,
                loss_type=config.loss_type,
                device=config.device,
            )

        # Create model instance (this loads the model from load_path)
        model = cls(config, from_scratch=False)

        return model

    @property
    def processor(self) -> Wav2Vec2Processor:
        """Get the model's processor for audio/text processing.

        Returns:
            Wav2Vec2 processor

        Examples:
            >>> processor = model.processor
            >>> text = processor.decode(token_ids)
        """
        return self._processor

    def get_collate_fn(
        self,
        languages: dict[str, str],
        packing_config: PackingConfig | None = None,
    ) -> Any:
        from flaime_serving.vendored.collate import create_ctc_collate_fn

        if packing_config is not None and packing_config.enabled:
            raise NotImplementedError(
                "Sample packing not yet implemented for CTC models (wav2vec2)."
            )
        return create_ctc_collate_fn(self._processor, languages)

    def get_expert_collate_fn(
        self,
        lang_code: str,
        packing_config: PackingConfig | None = None,
    ) -> Any:
        from flaime_serving.vendored.collate import create_ctc_expert_collate_fn

        if packing_config is not None and packing_config.enabled:
            raise NotImplementedError(
                "Sample packing not yet implemented for CTC models (wav2vec2)."
            )
        return create_ctc_expert_collate_fn(self._processor, lang_code)
