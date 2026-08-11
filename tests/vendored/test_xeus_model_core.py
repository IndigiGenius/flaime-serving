"""Tests for XEUS model wrapper.

This module tests the XEUSASRModel class that implements the BaseASRModel
interface for CMU WAVLab's XEUS (Cross-lingual Encoder for Universal Speech).

Task: 26Q1-XEUS-01 - XEUS Model Wrapper & ESPnet Integration
"""

from unittest.mock import Mock, patch

import pytest
import torch

from flaime_serving.vendored.base_model import BaseASRModelConfig


class TestXEUSModelConfig:
    """Test XEUS-specific configuration."""

    def test_xeus_config_defaults(self):
        """Test that XEUS config has correct default values."""
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
        )

        assert config.model_type == "xeus"
        assert config.sample_rate == 16000
        assert config.use_flash_attn is False
        assert config.loss_type == "ctc"

    def test_xeus_config_custom_values(self):
        """Test XEUS config with custom values."""
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            use_flash_attn=True,
            loss_type="multi",
            device="cuda",
        )

        assert config.use_flash_attn is True
        assert config.loss_type == "multi"
        assert config.device == "cuda"

    def test_xeus_config_intermediate_layers(self):
        """Test XEUS config with intermediate layer specification."""
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="multi",
            intermediate_layers=[12, 15, 18],
        )

        assert config.intermediate_layers == [12, 15, 18]


class TestXEUSASRModel:
    """Test XEUSASRModel implementation."""

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_model_initialization(self, mock_cuda, mock_load):
        """Test model initialization with default settings."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        # Setup mock XEUS model
        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Verify load_xeus_from_checkpoint was called
        mock_load.assert_called_once()

        # Verify model properties
        assert model.config == config
        assert model.xeus_model == mock_xeus_model

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_model_initialization_with_flash_attention(self, mock_cuda, mock_load):
        """Test model initialization with Flash Attention enabled.

        Standalone XEUS uses PyTorch SDPA which auto-selects Flash Attention.
        This test verifies model creation succeeds with use_flash_attn=True.
        """
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            use_flash_attn=True,
            device="cpu",
        )

        # Should not raise — Flash Attention is handled via PyTorch SDPA
        model = XEUSASRModel(config)
        assert model is not None

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_ctc_projection_layer_created(self, mock_cuda, mock_load):
        """Test that CTC projection layer is created for CTC loss."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="ctc",
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Verify CTC projection layer exists
        assert model.ctc_projection is not None
        assert isinstance(model.ctc_projection, torch.nn.Linear)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_returns_encoder_features(self, mock_cuda, mock_load):
        """Test forward pass returns encoder features and CTC logits."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        # Setup mock encoder output
        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        # encode() returns tuple: (features, lengths) when use_final_output=True
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Forward pass
        waveforms = torch.randn(batch_size, 16000)  # 1 second of audio
        wav_lengths = torch.tensor([16000, 16000])

        outputs = model(waveforms, wav_lengths=wav_lengths)

        # Verify output structure
        assert "logits" in outputs
        assert "encoder_features" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_with_labels_computes_loss(self, mock_cuda, mock_load):
        """Test forward pass with labels computes CTC loss."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Forward pass with labels
        waveforms = torch.randn(batch_size, 16000)
        wav_lengths = torch.tensor([16000, 16000])
        labels = torch.randint(1, 5000, (batch_size, 20))  # Avoid blank token 0

        outputs = model(waveforms, labels=labels, wav_lengths=wav_lengths)

        # Verify loss is computed
        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["loss"].dim() == 0  # Scalar

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_generate_returns_token_ids(self, mock_cuda, mock_load):
        """Test generate method returns predicted token IDs."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Generate
        waveforms = torch.randn(batch_size, 16000)

        with torch.no_grad():
            generated_ids = model.generate(waveforms)

        # Verify output
        assert isinstance(generated_ids, torch.Tensor)
        assert generated_ids.shape[0] == batch_size

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_derives_wav_lengths_from_attention_mask(
        self, mock_cuda, mock_load
    ):
        """Test that attention_mask is converted to wav_lengths when wav_lengths absent.

        The Trainer/loss dispatcher passes attention_mask (from collate) but
        XEUS expects wav_lengths. Without this conversion, padding is processed
        as real audio, causing NaN in the encoder for padded batches.
        """
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len // 2]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Simulate padded batch: sample 1 has 16000 real + 0 pad,
        # sample 2 has 8000 real + 8000 pad
        waveforms = torch.randn(batch_size, 16000)
        attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)
        attention_mask[1, 8000:] = 0  # Second sample is half-padded

        outputs = model(waveforms, attention_mask=attention_mask)

        # Verify encode was called with correct wav_lengths
        call_args = mock_xeus_model.encode.call_args
        wav_lengths_passed = call_args[0][1]  # Second positional arg
        assert wav_lengths_passed[0].item() == 16000
        assert wav_lengths_passed[1].item() == 8000

        assert "logits" in outputs

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_wav_lengths_takes_precedence_over_attention_mask(
        self, mock_cuda, mock_load
    ):
        """Test that explicit wav_lengths is used even when attention_mask is provided."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        waveforms = torch.randn(batch_size, 16000)
        wav_lengths = torch.tensor([16000, 12000])
        attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)

        outputs = model(
            waveforms, wav_lengths=wav_lengths, attention_mask=attention_mask
        )

        # wav_lengths should take precedence
        call_args = mock_xeus_model.encode.call_args
        wav_lengths_passed = call_args[0][1]
        assert wav_lengths_passed[0].item() == 16000
        assert wav_lengths_passed[1].item() == 12000

        assert "logits" in outputs

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_processor_property(self, mock_cuda, mock_load):
        """Test that processor property returns a valid processor."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Check processor exists
        processor = model.processor
        assert processor is not None

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_model_type_validation(self, mock_cuda, mock_load):
        """Test that model type must be 'xeus'."""
        from flaime_serving.vendored.xeus_model import XEUSASRModel

        mock_cuda.return_value = False

        # BaseASRModelConfig with wrong model type
        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="test",
            device="cpu",
        )

        with pytest.raises(ValueError, match="model_type='xeus'"):
            XEUSASRModel(config)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_attention_mask_converts_to_wav_lengths(self, mock_cuda, mock_load):
        """attention_mask is converted to wav_lengths for per-sample encoding."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Create attention_mask: sample 0 has 16000 valid frames, sample 1 has 8000
        waveforms = torch.randn(batch_size, 16000)
        attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)
        attention_mask[1, 8000:] = 0  # second sample is shorter

        outputs = model(waveforms, attention_mask=attention_mask)

        # Verify encode was called with per-sample wav_lengths (not full padded)
        call_args = mock_xeus_model.encode.call_args
        wav_lengths_arg = call_args[0][1]  # second positional arg
        assert wav_lengths_arg[0].item() == 16000
        assert wav_lengths_arg[1].item() == 8000

        assert "logits" in outputs

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_wav_lengths_takes_precedence(self, mock_cuda, mock_load):
        """Explicit wav_lengths is not overridden by attention_mask."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        waveforms = torch.randn(batch_size, 16000)
        wav_lengths = torch.tensor([12000, 10000])
        attention_mask = torch.ones(batch_size, 16000, dtype=torch.long)

        outputs = model(
            waveforms, wav_lengths=wav_lengths, attention_mask=attention_mask
        )

        # wav_lengths should be used, not attention_mask
        call_args = mock_xeus_model.encode.call_args
        wav_lengths_arg = call_args[0][1]
        assert wav_lengths_arg[0].item() == 12000
        assert wav_lengths_arg[1].item() == 10000
        assert "logits" in outputs
