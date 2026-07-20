"""XEUS ASR model implementation using BaseASRModel interface.

This module provides XEUSASRModel, a wrapper for CMU WAVLab's XEUS
(Cross-lingual Encoder for Universal Speech) model for ASR training.

XEUS is an E-Branchformer encoder trained on 1.1M hours across 4,057 languages,
making it ideal for indigenous and low-resource language ASR.

Key features:
- 577M parameter E-Branchformer encoder (19 layers)
- 16kHz audio input (raw waveforms)
- Flash Attention via PyTorch SDPA (auto-selects when available)
- CTC-based ASR with optional intermediate layer losses
- Standalone implementation (no ESPnet dependency)

Task: 26Q1-XEUS-01 - XEUS Model Wrapper
Reference: arXiv:2407.00837, HuggingFace: espnet/xeus
"""

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from flaime_serving.vendored.packing_config import PackingConfig

from .base_model import BaseASRModel, BaseASRModelConfig
from .xeus_standalone import StandaloneXEUS, load_xeus_from_checkpoint


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


# Default intermediate layers for XEUS (layers 12-19 are best for ASR)
DEFAULT_INTERMEDIATE_LAYERS = [12, 15, 18]

# XEUS architecture constants
XEUS_HIDDEN_SIZE = 1024
XEUS_NUM_LAYERS = 19


@dataclass
class XEUSModelConfig:
    """Configuration for XEUS ASR model.

    Standalone config for XEUS that provides BaseASRModelConfig-compatible attributes.

    Attributes:
        checkpoint_path: Path to XEUS checkpoint file (.pth)
        sample_rate: Expected audio sample rate (must be 16kHz for XEUS)
        use_flash_attn: Enable Flash Attention for efficient training
        vocab_size: Vocabulary size for CTC projection (default IPA ~150 tokens)
        intermediate_layers: Encoder layers for intermediate losses (multi-loss mode)
        freeze_encoder: Freeze entire encoder (for linear probe training)
        freeze_encoder_layers: Specific layers to freeze (partial fine-tuning)
        hidden_size: Encoder hidden dimension (default 1024 for XEUS)
        ctc_bottleneck_dim: If set, use a 2-layer MLP CTC head
            (hidden->bottleneck->ReLU->vocab) instead of a single Linear.
        loss_type: Type of loss function ("ctc" or "multi")
        device: Device to run the model on

    Examples:
        >>> config = XEUSModelConfig(
        ...     checkpoint_path="/path/to/xeus/checkpoint.pth",
        ...     use_flash_attn=True,
        ...     loss_type="multi",
        ...     intermediate_layers=[12, 15, 18],
        ... )
    """

    checkpoint_path: str
    sample_rate: int = 16000
    use_flash_attn: bool = False
    vocab_size: int = 5000
    intermediate_layers: list[int] | None = None
    freeze_encoder: bool = False
    freeze_encoder_layers: list[int] | None = None
    hidden_size: int = XEUS_HIDDEN_SIZE
    ctc_bottleneck_dim: int | None = None
    blank_bias_init: float | None = None
    loss_type: str = "ctc"
    device: str | None = None

    # These provide BaseASRModelConfig compatibility
    model_type: str = field(default="xeus", init=False)
    model_name_or_path: str = field(default="", init=False)

    def __post_init__(self):
        """Initialize device and set model_name_or_path."""
        if self.device is None:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set model_name_or_path from checkpoint_path
        object.__setattr__(self, "model_name_or_path", self.checkpoint_path)


class XEUSProcessor:
    """Simple processor for XEUS model.

    XEUS takes raw waveforms directly, so this processor mainly handles
    audio normalization and provides a consistent interface with other models.

    Attributes:
        sample_rate: Expected sample rate (16kHz)
    """

    def __init__(self, sample_rate: int = 16000):
        """Initialize processor.

        Args:
            sample_rate: Expected audio sample rate
        """
        self.sample_rate = sample_rate
        self.tokenizer: Any | None = None

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs from CTC decoding

        Returns:
            Decoded text string
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "XEUSProcessor has no tokenizer. Load a checkpoint via "
                "XEUSASRModel.from_pretrained() to initialise it."
            )
        return self.tokenizer.decode(token_ids)

    def __call__(
        self,
        audio: torch.Tensor | list,
        sampling_rate: int | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Process audio input.

        Args:
            audio: Raw audio waveform(s)
            sampling_rate: Audio sample rate (should match self.sample_rate)
            return_tensors: Return format (only 'pt' supported)

        Returns:
            Dictionary with 'input_values' tensor
        """
        if isinstance(audio, list):
            audio = torch.tensor(audio)
        elif not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)

        # Ensure float32
        if audio.dtype != torch.float32:
            audio = audio.float()

        # Normalize
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()

        # Ensure batch dimension (forward expects (batch, time))
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return {"input_values": audio}


class IntermediateCTCDecoders(nn.Module):
    """CTC projection heads for intermediate encoder layers.

    Used for multi-loss training with auxiliary CTC losses from
    intermediate encoder layers.

    Attributes:
        layers: List of layer indices to decode
        projections: ModuleDict mapping layer index to Linear projection
    """

    def __init__(self, layers: list[int], hidden_size: int, vocab_size: int):
        """Initialize intermediate decoders.

        Args:
            layers: Encoder layer indices for intermediate losses
            hidden_size: Encoder hidden dimension
            vocab_size: Output vocabulary size
        """
        super().__init__()
        self.layers = layers
        self.projections = nn.ModuleDict(
            {str(layer): nn.Linear(hidden_size, vocab_size) for layer in layers}
        )

    def forward(self, layer_outputs: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        """Project intermediate layer outputs to vocabulary logits.

        Args:
            layer_outputs: List of encoder layer outputs

        Returns:
            Dict mapping layer index to logits tensor for each intermediate layer
        """
        logits: dict[int, torch.Tensor] = {}
        for layer_idx in self.layers:
            if layer_idx < len(layer_outputs):
                proj = self.projections[str(layer_idx)]
                logits[layer_idx] = proj(layer_outputs[layer_idx])
        return logits


class XEUSASRModel(BaseASRModel):
    """XEUS model wrapper for CTC-based ASR training.

    This class implements the BaseASRModel interface for XEUS,
    providing CTC-based ASR with optional intermediate layer losses
    for multi-loss training.

    XEUS is an encoder-only model, so it does NOT support RNN-T.

    Attributes:
        config: Model configuration
        xeus_model: Underlying ESPnet XEUS model
        ctc_projection: CTC projection layer (hidden_size -> vocab_size)
        intermediate_decoders: Optional intermediate layer CTC heads
        rnnt_decoder: Always None (XEUS is encoder-only)
        _processor: Audio processor

    Examples:
        >>> config = XEUSModelConfig(
        ...     checkpoint_path="/path/to/xeus/checkpoint.pth",
        ...     loss_type="ctc",
        ... )
        >>> model = XEUSASRModel(config)
        >>> outputs = model(waveforms, wav_lengths=lengths)
    """

    def __init__(
        self,
        config: XEUSModelConfig | BaseASRModelConfig,
        from_scratch: bool = False,
    ) -> None:
        """Initialize XEUS model.

        Args:
            config: Model configuration
            from_scratch: If True, initialize with random weights instead of
                loading the SSL checkpoint. Useful for controlled experiments
                comparing architecture vs. pretraining effects.

        Raises:
            ValueError: If model_type is not 'xeus'
        """
        super().__init__()

        # Validate model type
        if config.model_type.lower() != "xeus":
            raise ValueError(
                f"XEUSASRModel requires model_type='xeus', got '{config.model_type}'"
            )

        # Store config (handle both config types)
        self.config: XEUSModelConfig | BaseASRModelConfig
        if isinstance(config, XEUSModelConfig):
            self.config = config
            checkpoint_path = config.checkpoint_path
            use_flash_attn = config.use_flash_attn
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            intermediate_layers = config.intermediate_layers
            freeze_encoder = config.freeze_encoder
            freeze_encoder_layers = config.freeze_encoder_layers
        else:
            # BaseASRModelConfig - use defaults
            self.config = config
            checkpoint_path = config.model_name_or_path
            use_flash_attn = False
            vocab_size = 5000
            hidden_size = XEUS_HIDDEN_SIZE
            intermediate_layers = None
            freeze_encoder = False
            freeze_encoder_layers = None

        # Initialize XEUS encoder
        if from_scratch:
            # Random weights — no SSL checkpoint loaded
            self.xeus_model: StandaloneXEUS = StandaloneXEUS()
            print("Initialized XEUS model with RANDOM weights (from_scratch=True)")
        else:
            # Load SSL checkpoint (standalone, no ESPnet dependency)
            self.xeus_model = load_xeus_from_checkpoint(
                checkpoint_path, device=config.device or "cpu"
            )

        # Flash Attention: standalone implementation uses PyTorch SDPA
        # which auto-selects Flash Attention when available (PyTorch 2.0+)
        if use_flash_attn:
            print("Flash Attention enabled via PyTorch scaled_dot_product_attention")

        # CTC head: optional bottleneck MLP for SSL feature rank collapse.
        # SSL-pretrained XEUS features live in a ~350/1024-dim subspace;
        # a bottleneck Linear(1024,384)->ReLU->Linear(384,vocab) forces
        # the projection to learn a low-rank mapping matching that geometry.
        ctc_bottleneck_dim = (
            config.ctc_bottleneck_dim if isinstance(config, XEUSModelConfig) else None
        )
        self.ctc_norm: nn.LayerNorm | None = None
        self.ctc_bottleneck: nn.Linear | None = None
        self.ctc_projection: nn.Linear | None = None
        if config.loss_type in ("ctc", "multi"):
            self.ctc_norm = nn.LayerNorm(hidden_size)
            if ctc_bottleneck_dim is not None:
                self.ctc_bottleneck = nn.Linear(hidden_size, ctc_bottleneck_dim)
                self.ctc_projection = nn.Linear(ctc_bottleneck_dim, vocab_size)
                if _is_main_process():
                    print(
                        f"Added CTC head: LayerNorm({hidden_size}) -> "
                        f"{hidden_size} -> {ctc_bottleneck_dim} -> {vocab_size}"
                    )
            else:
                self.ctc_projection = nn.Linear(hidden_size, vocab_size)
                if _is_main_process():
                    print(
                        f"Added CTC head: LayerNorm({hidden_size}) -> "
                        f"{hidden_size} -> {vocab_size}"
                    )

            # Suppress blank token logit to counter CTC blank-collapse attractor
            blank_bias = (
                config.blank_bias_init if isinstance(config, XEUSModelConfig) else None
            )
            if blank_bias is not None and self.ctc_projection is not None:
                with torch.no_grad():
                    self.ctc_projection.bias.data[0] = blank_bias
                if _is_main_process():
                    print(f"Set blank bias init: ctc_projection.bias[0] = {blank_bias}")

        # Create intermediate decoders for multi-loss
        self.intermediate_decoders: IntermediateCTCDecoders | None = None
        if config.loss_type == "multi":
            layers = intermediate_layers or DEFAULT_INTERMEDIATE_LAYERS
            self.intermediate_decoders = IntermediateCTCDecoders(
                layers=layers,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
            )
            print(f"Added intermediate CTC decoders for layers: {layers}")

        # XEUS is encoder-only, no RNN-T support
        self.rnnt_decoder = None

        # Create processor
        sample_rate = (
            config.sample_rate if isinstance(config, XEUSModelConfig) else 16000
        )
        self._processor = XEUSProcessor(sample_rate=sample_rate)

        # Apply encoder freezing
        if freeze_encoder:
            self.freeze_encoder()
        elif freeze_encoder_layers:
            self.freeze_encoder_layers(freeze_encoder_layers)

        # Move to device
        self.to(config.device)

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (for linear probe training)."""
        for param in self.xeus_model.encoder.parameters():
            param.requires_grad = False
        print("Froze all XEUS encoder parameters")

    def freeze_encoder_layers(self, layers: list[int]) -> None:
        """Freeze specific encoder layers.

        Args:
            layers: List of layer indices to freeze (0-indexed)
        """
        for layer_idx in layers:
            if layer_idx < len(self.xeus_model.encoder.encoders):
                for param in self.xeus_model.encoder.encoders[layer_idx].parameters():
                    param.requires_grad = False
        print(f"Froze XEUS encoder layers: {layers}")

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        wav_lengths: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Forward pass through XEUS encoder.

        Args:
            input_features: Raw audio waveforms, shape (batch, time)
            labels: Optional target labels for CTC loss, shape (batch, seq_len)
            wav_lengths: Optional waveform lengths, shape (batch,)
            attention_mask: Optional padding mask, shape (batch, time).
                Used to derive wav_lengths when wav_lengths is not provided.
                This is the standard interface from the Trainer/loss dispatcher.
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary containing:
                - 'logits': CTC logits, shape (batch, time, vocab_size)
                - 'ctc_logits': Alias for logits (MultiLoss compatibility)
                - 'encoder_features': Final encoder output
                - 'encoder_hidden_states': Alias for encoder_features (standardized key)
                - 'input_lengths': Output sequence lengths, shape (batch,)
                - 'loss': CTC loss (only if labels provided)
                - 'intermediate_logits': Dict mapping layer index to logits (multi-loss)
        """
        batch_size = input_features.shape[0]

        # Derive wav_lengths from attention_mask if not explicitly provided.
        # The Trainer passes attention_mask (from collate) but XEUS needs
        # wav_lengths for correct frontend layer_norm and encoder output lengths.
        if wav_lengths is None and attention_mask is not None:
            wav_lengths = attention_mask.sum(dim=1).long()
        elif wav_lengths is None:
            wav_lengths = torch.full(
                (batch_size,), input_features.shape[1], device=input_features.device
            )

        # Get encoder outputs (all layers for multi-loss)
        use_final_only = self.intermediate_decoders is None
        layer_outputs, output_lengths = self.xeus_model.encode(
            input_features,
            wav_lengths,
            use_mask=False,
            use_final_output=use_final_only,
        )

        # Get final layer features
        if use_final_only:
            # layer_outputs is the final output directly
            encoder_features = layer_outputs
        else:
            # layer_outputs is a list, take the last one
            encoder_features = layer_outputs[-1]

        # CTC head: LayerNorm (scale fix) + optional bottleneck MLP (rank fix)
        if self.ctc_norm is not None:
            encoder_features = self.ctc_norm(encoder_features)
        if self.ctc_bottleneck is not None:
            encoder_features = F.relu(self.ctc_bottleneck(encoder_features))
        logits = self.ctc_projection(encoder_features)

        # Build result with standardized keys for Trainer compatibility
        # - "logits": Primary output (used by generate, CTC loss)
        # - "ctc_logits": Alias for multi-loss framework compatibility
        # - "encoder_hidden_states": Standardized name for encoder output
        # - "input_lengths": Output sequence lengths (avoids hardcoded downsampling)
        result: dict[str, Any] = {
            "logits": logits,
            "ctc_logits": logits,  # Alias for MultiLoss compatibility
            "encoder_features": encoder_features,
            "encoder_hidden_states": encoder_features,  # Standardized key
            "input_lengths": output_lengths,  # For Trainer to compute CTC lengths
        }

        # Intermediate logits for multi-loss
        if self.intermediate_decoders is not None and not use_final_only:
            result["intermediate_logits"] = self.intermediate_decoders(layer_outputs)

        # Compute CTC loss if labels provided
        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (time, batch, vocab)

            # Get input and target lengths
            input_lengths = output_lengths
            target_lengths = (labels != -100).sum(dim=-1)

            # Replace -100 with 0 for CTC (will be masked by target_lengths)
            labels_for_ctc = labels.clone()
            labels_for_ctc[labels_for_ctc == -100] = 0

            loss = F.ctc_loss(
                log_probs,
                labels_for_ctc,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            result["loss"] = loss

        return result

    def generate(self, input_features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate transcriptions via CTC greedy decoding.

        Args:
            input_features: Raw audio waveforms, shape (batch, time)
            **kwargs: Additional arguments (wav_lengths, etc.)

        Returns:
            Predicted token IDs, shape (batch, time)
        """
        wav_lengths = kwargs.get("wav_lengths")

        with torch.no_grad():
            outputs = self.forward(input_features, wav_lengths=wav_lengths)
            logits = outputs["logits"]

        # Greedy decoding: argmax
        predicted_ids = torch.argmax(logits, dim=-1)

        return predicted_ids

    def save_pretrained(self, save_path: str) -> None:
        """Save model state to disk.

        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)

        # Save full state dict (XEUS + CTC projection + intermediate decoders)
        state_dict = {
            "xeus_model": self.xeus_model.state_dict(),
            "ctc_norm": (
                self.ctc_norm.state_dict() if self.ctc_norm is not None else None
            ),
            "ctc_bottleneck": (
                self.ctc_bottleneck.state_dict()
                if self.ctc_bottleneck is not None
                else None
            ),
            "ctc_projection": (
                self.ctc_projection.state_dict()
                if self.ctc_projection is not None
                else None
            ),
            "intermediate_decoders": (
                self.intermediate_decoders.state_dict()
                if self.intermediate_decoders is not None
                else None
            ),
            "config": {
                "model_type": self.config.model_type,
                "loss_type": self.config.loss_type,
                "device": str(self.config.device),
            },
        }

        torch.save(state_dict, os.path.join(save_path, "xeus_asr_model.pt"))
        print(f"Saved XEUS ASR model to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        config: XEUSModelConfig | BaseASRModelConfig | None = None,
    ) -> "XEUSASRModel":
        """Load model from saved checkpoint.

        Args:
            load_path: Path to saved model directory or checkpoint
            config: Optional configuration override

        Returns:
            Loaded model instance
        """
        if config is None:
            config = XEUSModelConfig(checkpoint_path=load_path)

        # FLAIME training checkpoints (.pt files with a model_state_dict key)
        # hold the full XEUSASRModel state dict produced by the trainer or the
        # BTM merge sweep. Detect this format and load directly rather than
        # going through the two-step SSL-init + ASR-head-load path, which is
        # only valid for raw HuggingFace XEUS SSL checkpoints.
        if os.path.isfile(load_path) and load_path.endswith(".pt"):
            # weights_only=False: full checkpoint pickle written by this repo's
            # trainer; trust the source — FLAIME-internal artifact.
            ckpt = torch.load(load_path, map_location=config.device, weights_only=False)  # noqa: S614
            if "model_state_dict" in ckpt:
                from flaime_serving.vendored.ctc_vocab import (
                    maybe_resize_ctc_head_from_state_dict,
                )

                raw = ckpt["model_state_dict"]
                state_dict = {
                    (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
                    for k, v in raw.items()
                }
                model = cls(config, from_scratch=True)

                # Pre-2026-04-16 checkpoints lack ctc_norm / ctc_bottleneck
                if (
                    getattr(model, "ctc_norm", None) is not None
                    and "ctc_norm.weight" not in state_dict
                ):
                    model.ctc_norm = None
                if (
                    getattr(model, "ctc_bottleneck", None) is not None
                    and "ctc_bottleneck.weight" not in state_dict
                ):
                    model.ctc_bottleneck = None

                maybe_resize_ctc_head_from_state_dict(model, state_dict)
                model.load_state_dict(state_dict)

                # Load vocab from ctc_vocab.json next to the checkpoint
                from flaime_serving.vendored.ctc_vocab import load_ctc_tokenizer

                vocab_path = os.path.join(
                    os.path.dirname(os.path.abspath(load_path)), "ctc_vocab.json"
                )
                if os.path.exists(vocab_path):
                    model._processor.tokenizer = load_ctc_tokenizer(vocab_path)
                else:
                    print(
                        f"Warning: ctc_vocab.json not found at {vocab_path}. "
                        "processor.decode() will not work."
                    )

                return model

        # HuggingFace SSL checkpoint path: init encoder from SSL weights, then
        # optionally overlay the ASR head from a sidecar xeus_asr_model.pt.
        model = cls(config)

        state_path = os.path.join(load_path, "xeus_asr_model.pt")
        if os.path.exists(state_path):
            # weights_only=True is safe: save() writes only nested state
            # dicts, None placeholders, and primitive config strings.
            state_dict = torch.load(
                state_path, map_location=config.device, weights_only=True
            )

            if state_dict.get("ctc_norm") and model.ctc_norm is not None:
                model.ctc_norm.load_state_dict(state_dict["ctc_norm"])

            if state_dict.get("ctc_bottleneck") and model.ctc_bottleneck is not None:
                model.ctc_bottleneck.load_state_dict(state_dict["ctc_bottleneck"])

            if state_dict.get("ctc_projection") and model.ctc_projection is not None:
                model.ctc_projection.load_state_dict(state_dict["ctc_projection"])

            if (
                state_dict.get("intermediate_decoders")
                and model.intermediate_decoders is not None
            ):
                model.intermediate_decoders.load_state_dict(
                    state_dict["intermediate_decoders"]
                )

            print(f"Loaded XEUS ASR model state from {state_path}")

        return model

    @property
    def processor(self) -> XEUSProcessor:
        """Get the model's audio processor.

        Returns:
            XEUSProcessor instance
        """
        return self._processor

    def get_collate_fn(
        self,
        languages: dict[str, str],
        packing_config: PackingConfig | None = None,
    ) -> Any:
        from flaime_serving.vendored.collate import create_espnet_collate_fn

        if packing_config is not None and packing_config.enabled:
            raise NotImplementedError(
                "Sample packing not yet implemented for ESPnet models (xeus)."
            )
        return create_espnet_collate_fn(self, languages)

    def get_expert_collate_fn(
        self,
        lang_code: str,
        packing_config: PackingConfig | None = None,
    ) -> Any:
        from flaime_serving.vendored.collate import create_espnet_expert_collate_fn

        if packing_config is not None and packing_config.enabled:
            raise NotImplementedError(
                "Sample packing not yet implemented for ESPnet models (xeus)."
            )
        return create_espnet_expert_collate_fn(self, lang_code)
