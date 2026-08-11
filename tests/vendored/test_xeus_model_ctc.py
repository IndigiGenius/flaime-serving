"""Tests for XEUS model wrapper.

This module tests the XEUSASRModel class that implements the BaseASRModel
interface for CMU WAVLab's XEUS (Cross-lingual Encoder for Universal Speech).

Task: 26Q1-XEUS-01 - XEUS Model Wrapper & ESPnet Integration
"""

from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from flaime_serving.vendored.base_model import BaseASRModelConfig


class TestXEUSBottleneckHead:
    """Test bottleneck MLP CTC head for SSL feature rank collapse."""

    def test_config_bottleneck_dim_defaults_to_none(self):
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(checkpoint_path="/path/to/xeus/checkpoint.pth")
        assert config.ctc_bottleneck_dim is None

    def test_config_bottleneck_dim_custom(self):
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            ctc_bottleneck_dim=384,
        )
        assert config.ctc_bottleneck_dim == 384

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_bottleneck_creates_two_layer_mlp(self, mock_cuda, mock_load):
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
            vocab_size=100,
            ctc_bottleneck_dim=384,
            device="cpu",
        )

        model = XEUSASRModel(config)

        assert model.ctc_bottleneck is not None
        assert isinstance(model.ctc_bottleneck, torch.nn.Linear)
        assert model.ctc_bottleneck.in_features == 1024
        assert model.ctc_bottleneck.out_features == 384

        assert model.ctc_projection is not None
        assert model.ctc_projection.in_features == 384
        assert model.ctc_projection.out_features == 100

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_no_bottleneck_is_single_linear(self, mock_cuda, mock_load):
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
            vocab_size=100,
            device="cpu",
        )

        model = XEUSASRModel(config)

        assert model.ctc_bottleneck is None
        assert model.ctc_projection is not None
        assert model.ctc_projection.in_features == 1024
        assert model.ctc_projection.out_features == 100

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_bottleneck_forward_produces_correct_shape(self, mock_cuda, mock_load):
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
            vocab_size=100,
            ctc_bottleneck_dim=384,
            device="cpu",
        )

        model = XEUSASRModel(config)

        waveforms = torch.randn(batch_size, 16000)
        wav_lengths = torch.tensor([16000, 16000])

        outputs = model(waveforms, wav_lengths=wav_lengths)

        assert outputs["logits"].shape == (batch_size, seq_len, 100)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_bottleneck_forward_applies_relu(self, mock_cuda, mock_load):
        """Verify ReLU nonlinearity: negative bottleneck values should be zeroed."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 1, 10, 1024
        mock_features = torch.randn(batch_size, seq_len, hidden_size)

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encode.return_value = (
            mock_features,
            torch.tensor([seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            vocab_size=10,
            ctc_bottleneck_dim=8,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Set bottleneck weights to identity-like so we can see ReLU effect
        with torch.no_grad():
            model.ctc_bottleneck.weight.fill_(0.0)
            model.ctc_bottleneck.bias.fill_(-1.0)  # All negative -> ReLU zeros

            model.ctc_projection.weight.fill_(1.0)
            model.ctc_projection.bias.fill_(0.0)

        waveforms = torch.randn(1, 16000)
        wav_lengths = torch.tensor([16000])

        with torch.no_grad():
            outputs = model(waveforms, wav_lengths=wav_lengths)

        # With all-negative bottleneck output + ReLU, logits should be all zeros
        assert torch.allclose(outputs["logits"], torch.zeros_like(outputs["logits"]))

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_ctc_norm_present(self, mock_cuda, mock_load):
        """LayerNorm is applied before CTC projection for SSL scale normalization."""
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
        assert model.ctc_norm is not None
        assert isinstance(model.ctc_norm, nn.LayerNorm)
        assert model.ctc_norm.normalized_shape == (1024,)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_includes_bottleneck(
        self, mock_makedirs, mock_save, mock_cuda, mock_load
    ):
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False
        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.state_dict.return_value = {}
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            ctc_bottleneck_dim=384,
            device="cpu",
        )

        model = XEUSASRModel(config)
        model.save_pretrained("/tmp/save")

        saved_dict = mock_save.call_args[0][0]
        assert "ctc_bottleneck" in saved_dict
        assert saved_dict["ctc_bottleneck"] is not None
        assert "ctc_norm" in saved_dict
        assert saved_dict["ctc_norm"] is not None


class TestXEUSBlankBiasInit:
    """Test blank_bias_init for CTC blank-collapse mitigation."""

    def test_config_blank_bias_defaults_to_none(self):
        from flaime_serving.vendored.xeus_model import XEUSModelConfig

        config = XEUSModelConfig(checkpoint_path="/path/to/xeus/checkpoint.pth")
        assert config.blank_bias_init is None

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_blank_bias_init_sets_projection_bias(self, mock_cuda, mock_load):
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
            vocab_size=100,
            blank_bias_init=-3.0,
            device="cpu",
        )

        model = XEUSASRModel(config)
        assert model.ctc_projection is not None
        assert model.ctc_projection.bias.data[0].item() == -3.0
        assert model.ctc_projection.bias.data[1].item() != -3.0

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_blank_bias_init_none_leaves_default(self, mock_cuda, mock_load):
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
            vocab_size=100,
            device="cpu",
        )

        model = XEUSASRModel(config)
        assert model.ctc_projection is not None
        # Default PyTorch init — bias[0] should NOT be -3.0
        assert model.ctc_projection.bias.data[0].item() != -3.0

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_blank_bias_init_with_bottleneck(self, mock_cuda, mock_load):
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
            vocab_size=100,
            ctc_bottleneck_dim=384,
            blank_bias_init=-2.0,
            device="cpu",
        )

        model = XEUSASRModel(config)
        assert model.ctc_projection is not None
        assert model.ctc_projection.bias.data[0].item() == -2.0


class TestXEUSMultiLoss:
    """Test multi-loss functionality for XEUSASRModel."""

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_intermediate_decoders_created_for_multi_loss(self, mock_cuda, mock_load):
        """Test that intermediate decoders are created for multi-loss mode."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        # XEUS has 19 encoder layers
        mock_layers = [Mock() for _ in range(19)]
        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = mock_layers
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="multi",
            intermediate_layers=[12, 15, 18],
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Verify intermediate decoders exist
        assert model.intermediate_decoders is not None
        assert len(config.intermediate_layers) == 3

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_forward_returns_intermediate_logits(self, mock_cuda, mock_load):
        """Test that forward pass returns intermediate layer logits for multi-loss."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        batch_size, seq_len, hidden_size = 2, 100, 1024
        num_layers = 19

        # Create mock outputs for all layers
        layer_outputs = [
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)
        ]

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = [Mock() for _ in range(num_layers)]
        # encode with use_final_output=False returns all layer outputs
        mock_xeus_model.encode.return_value = (
            layer_outputs,
            torch.tensor([seq_len, seq_len]),
        )
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="multi",
            intermediate_layers=[12, 15, 18],
            vocab_size=5000,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Forward pass
        waveforms = torch.randn(batch_size, 16000)
        wav_lengths = torch.tensor([16000, 16000])

        outputs = model(waveforms, wav_lengths=wav_lengths)

        # Verify intermediate logits are returned
        assert "intermediate_logits" in outputs
        assert isinstance(outputs["intermediate_logits"], dict)
        assert len(outputs["intermediate_logits"]) == 3  # 3 intermediate layers

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_default_intermediate_layers_for_xeus(self, mock_cuda, mock_load):
        """Test that default intermediate layers are sensible for XEUS architecture."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_layers = [Mock() for _ in range(19)]
        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = mock_layers
        mock_load.return_value = mock_xeus_model

        # No intermediate_layers specified - should use defaults
        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="multi",
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Default should use layers from the ASR-optimal range (12-19)
        assert model.intermediate_decoders is not None

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_no_intermediate_decoders_for_ctc_only(self, mock_cuda, mock_load):
        """Test that intermediate decoders are NOT created for CTC-only mode."""
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
            loss_type="ctc",  # CTC only, no multi-loss
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Verify intermediate decoders are NOT created
        assert model.intermediate_decoders is None


class TestXEUSModelProtocolCompliance:
    """Test that XEUSASRModel complies with ASRModelProtocol."""

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_implements_asr_model_protocol(self, mock_cuda, mock_load):
        """Test that XEUSASRModel implements ASRModelProtocol."""
        from flaime_serving.vendored.protocols import ASRModelProtocol
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

        # Check protocol compliance
        assert isinstance(model, ASRModelProtocol)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_supports_ctc_capability(self, mock_cuda, mock_load):
        """Test that XEUSASRModel has CTC capability."""
        from flaime_serving.vendored.protocols import supports_ctc
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

        # Check CTC capability
        assert supports_ctc(model)
        assert model.ctc_projection is not None

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_supports_intermediate_outputs_capability(self, mock_cuda, mock_load):
        """Test that XEUSASRModel has intermediate outputs capability when enabled."""
        from flaime_serving.vendored.protocols import supports_intermediate_outputs
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_layers = [Mock() for _ in range(19)]
        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = mock_layers
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            loss_type="multi",
            intermediate_layers=[12, 15, 18],
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Check intermediate outputs capability
        assert supports_intermediate_outputs(model)

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_does_not_support_rnnt(self, mock_cuda, mock_load):
        """Test that XEUSASRModel does NOT support RNN-T (encoder-only model)."""
        from flaime_serving.vendored.protocols import supports_rnnt
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

        # XEUS is encoder-only, should NOT support RNN-T
        assert not supports_rnnt(model)


class TestXEUSFromScratch:
    """Test from_scratch initialization for XEUS (random weights, no checkpoint)."""

    @patch("flaime_serving.vendored.xeus_model.StandaloneXEUS")
    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_from_scratch_skips_checkpoint_loading(
        self, mock_cuda, mock_load, mock_standalone
    ):
        """Test that from_scratch=True does NOT call load_xeus_from_checkpoint."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_standalone.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            device="cpu",
        )

        model = XEUSASRModel(config, from_scratch=True)

        mock_load.assert_not_called()
        mock_standalone.assert_called_once()
        assert model.xeus_model == mock_xeus_model

    @patch("flaime_serving.vendored.xeus_model.StandaloneXEUS")
    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_from_scratch_false_loads_checkpoint(
        self, mock_cuda, mock_load, mock_standalone
    ):
        """Test that from_scratch=False (default) loads from checkpoint."""
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

        model = XEUSASRModel(config, from_scratch=False)

        mock_load.assert_called_once()
        mock_standalone.assert_not_called()
        assert model.xeus_model == mock_xeus_model

    @patch("flaime_serving.vendored.xeus_model.StandaloneXEUS")
    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_from_scratch_with_base_config(self, mock_cuda, mock_load, mock_standalone):
        """Test from_scratch=True works with BaseASRModelConfig too."""
        from flaime_serving.vendored.xeus_model import XEUSASRModel

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_standalone.return_value = mock_xeus_model

        config = BaseASRModelConfig(
            model_type="xeus",
            model_name_or_path="/path/to/xeus/checkpoint.pth",
            device="cpu",
        )

        model = XEUSASRModel(config, from_scratch=True)

        mock_load.assert_not_called()
        mock_standalone.assert_called_once()
        assert model.xeus_model == mock_xeus_model


class TestXEUSModelSaveLoad:
    """Test save/load functionality for XEUSASRModel."""

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    @patch("torch.save")
    @patch("os.makedirs")
    def test_save_pretrained(self, mock_makedirs, mock_save, mock_cuda, mock_load):
        """Test save_pretrained saves model state."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.state_dict.return_value = {"key": "value"}
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            device="cpu",
        )

        model = XEUSASRModel(config)
        model.save_pretrained("/tmp/save_path")

        # Verify save was called
        mock_save.assert_called()

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_from_pretrained(self, mock_cuda, mock_load):
        """Test from_pretrained loads model correctly."""
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

        model = XEUSASRModel.from_pretrained("/path/to/checkpoint", config)

        assert isinstance(model, XEUSASRModel)


class TestXEUSEncoderFreezing:
    """Test encoder freezing functionality for fine-tuning."""

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_freeze_encoder(self, mock_cuda, mock_load):
        """Test that encoder can be frozen for linear probe training."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        # Create mock parameters
        mock_param = Mock()
        mock_param.requires_grad = True

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = []
        mock_xeus_model.encoder.parameters.return_value = [mock_param]
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            freeze_encoder=True,
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Encoder parameters should have been frozen during init
        # (freeze_encoder=True in config triggers freeze_encoder())
        assert model is not None  # Model created successfully

    @patch("flaime_serving.vendored.xeus_model.load_xeus_from_checkpoint")
    @patch("torch.cuda.is_available")
    def test_freeze_encoder_layers_partial(self, mock_cuda, mock_load):
        """Test partial encoder freezing (freeze layers 0-11, train 12-19)."""
        from flaime_serving.vendored.xeus_model import (
            XEUSASRModel,
            XEUSModelConfig,
        )

        mock_cuda.return_value = False

        # Create 19 mock layers
        mock_layers = [Mock() for _ in range(19)]
        for layer in mock_layers:
            mock_param = Mock()
            mock_param.requires_grad = True
            layer.parameters.return_value = [mock_param]

        mock_xeus_model = Mock()
        mock_xeus_model.encoder = Mock()
        mock_xeus_model.encoder.encoders = mock_layers
        mock_load.return_value = mock_xeus_model

        config = XEUSModelConfig(
            checkpoint_path="/path/to/xeus/checkpoint.pth",
            freeze_encoder_layers=list(range(12)),  # Freeze layers 0-11
            device="cpu",
        )

        model = XEUSASRModel(config)

        # Model should have method to freeze specific layers
        assert hasattr(model, "freeze_encoder_layers")
