"""Tests for Wav2Vec2 model wrapper.

This module tests the Wav2Vec2ASRModel class that implements the BaseASRModel
interface for Facebook's Wav2Vec2 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from flaime_serving.vendored.base_model import BaseASRModelConfig
from flaime_serving.vendored.wav2vec2_model import Wav2Vec2ASRModel

_MODULE = "flaime_serving.vendored.wav2vec2_model"


@dataclass
class Wav2Vec2Mocks:
    """Container for common Wav2Vec2 test mocks."""

    config_class: Mock
    wav2vec2_config: Mock
    model_class: Mock
    model_instance: Mock
    processor_class: Mock
    processor_instance: Mock
    _patches: list[Any] = field(default_factory=list, repr=False)

    def stop_all(self) -> None:
        for p in self._patches:
            p.stop()


def _create_wav2vec2_mocks(
    architectures: list[str] | None = None,
) -> Wav2Vec2Mocks:
    """Create and start the standard wav2vec2 mock set."""
    if architectures is None:
        architectures = ["Wav2Vec2ForCTC"]

    patches = [
        patch("torch.cuda.is_available", return_value=False),
        patch(f"{_MODULE}.Wav2Vec2ForCTC"),
        patch(f"{_MODULE}.Wav2Vec2Processor"),
        patch(f"{_MODULE}.Wav2Vec2Config"),
    ]
    started = [p.start() for p in patches]
    mock_cuda, mock_model_class, mock_processor_class, mock_config_class = started

    # Wav2Vec2Config.from_pretrained → config with architectures
    mock_wav2vec2_config = Mock(
        architectures=architectures, hidden_size=768, vocab_size=32
    )
    mock_config_class.from_pretrained.return_value = mock_wav2vec2_config

    # Wav2Vec2ForCTC (constructor and from_pretrained)
    mock_model_instance = Mock()
    mock_model_instance.config = Mock(hidden_size=768, vocab_size=32)
    mock_model_instance.to = Mock(return_value=mock_model_instance)
    mock_model_class.from_pretrained.return_value = mock_model_instance
    mock_model_class.return_value = mock_model_instance

    # Wav2Vec2Processor
    mock_processor_instance = Mock()
    mock_processor_class.from_pretrained.return_value = mock_processor_instance

    return Wav2Vec2Mocks(
        config_class=mock_config_class,
        wav2vec2_config=mock_wav2vec2_config,
        model_class=mock_model_class,
        model_instance=mock_model_instance,
        processor_class=mock_processor_class,
        processor_instance=mock_processor_instance,
        _patches=patches,
    )


@pytest.fixture
def wav2vec2_mocks():
    """Patch Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor, and CUDA."""
    mocks = _create_wav2vec2_mocks()
    yield mocks
    mocks.stop_all()


@pytest.fixture
def wav2vec2_ssl_mocks():
    """Same as wav2vec2_mocks but config reports an SSL-only architecture."""
    mocks = _create_wav2vec2_mocks(architectures=["Wav2Vec2ForPreTraining"])
    yield mocks
    mocks.stop_all()


def _make_config(model_name: str = "facebook/wav2vec2-base") -> BaseASRModelConfig:
    return BaseASRModelConfig(
        model_type="wav2vec2",
        model_name_or_path=model_name,
        loss_type="ctc",
        device="cpu",
    )


class TestWav2Vec2ASRModel:
    """Test Wav2Vec2ASRModel implementation."""

    def test_model_initialization_default(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test model initialization with default settings."""
        m = wav2vec2_mocks
        config = _make_config()
        model = Wav2Vec2ASRModel(config)

        m.model_class.from_pretrained.assert_called_once_with("facebook/wav2vec2-base")
        m.processor_class.from_pretrained.assert_called_once_with(
            "facebook/wav2vec2-base"
        )
        assert model.config == config
        assert model.model == m.model_instance

    def test_model_initialization_from_scratch(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test model initialization from scratch (random weights)."""
        m = wav2vec2_mocks
        config = _make_config()
        model = Wav2Vec2ASRModel(config, from_scratch=True)

        m.config_class.from_pretrained.assert_called_once_with("facebook/wav2vec2-base")
        m.model_class.assert_called_once_with(m.wav2vec2_config)
        assert model.model == m.model_instance

    def test_forward_with_labels(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test forward pass with labels (training mode)."""
        m = wav2vec2_mocks
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.5)
        mock_output.logits = torch.randn(2, 100, 32)
        m.model_instance.return_value = mock_output

        model = Wav2Vec2ASRModel(_make_config())
        input_features = torch.randn(2, 16000)
        labels = torch.randint(0, 32, (2, 50))
        outputs = model(input_features, labels)

        assert "loss" in outputs
        assert "logits" in outputs
        assert torch.equal(outputs["loss"], mock_output.loss)
        assert torch.equal(outputs["logits"], mock_output.logits)

        m.model_instance.assert_called_once()
        call_args = m.model_instance.call_args
        assert torch.equal(call_args[1]["input_values"], input_features)
        assert torch.equal(call_args[1]["labels"], labels)

    def test_forward_without_labels(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test forward pass without labels (inference mode)."""
        m = wav2vec2_mocks
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 100, 32)
        mock_output.loss = None
        m.model_instance.return_value = mock_output

        model = Wav2Vec2ASRModel(_make_config())
        input_features = torch.randn(2, 16000)
        outputs = model(input_features)

        assert "logits" in outputs
        assert torch.equal(outputs["logits"], mock_output.logits)
        assert "loss" not in outputs or outputs["loss"] is None

    def test_forward_always_returns_dict_with_ctc_logits(
        self, wav2vec2_mocks: Wav2Vec2Mocks
    ):
        """Test forward always returns dict with logits and ctc_logits keys."""
        m = wav2vec2_mocks
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 100, 32)
        mock_output.loss = None
        m.model_instance.return_value = mock_output

        model = Wav2Vec2ASRModel(_make_config())
        result = model(torch.randn(2, 16000))

        assert isinstance(result, dict)
        assert "logits" in result
        assert "ctc_logits" in result
        assert torch.equal(result["logits"], mock_output.logits)
        assert torch.equal(result["ctc_logits"], mock_output.logits)

    def test_generate(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test generate method for transcription."""
        m = wav2vec2_mocks
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 100, 32)
        m.model_instance.return_value = mock_output
        m.processor_instance.batch_decode.return_value = [
            "hello world",
            "test transcription",
        ]

        model = Wav2Vec2ASRModel(_make_config())
        generated_ids = model.generate(torch.randn(2, 16000))

        m.model_instance.assert_called()
        assert isinstance(generated_ids, torch.Tensor)
        assert generated_ids.shape[0] == 2

    def test_processor_property(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test that processor property returns correct processor."""
        m = wav2vec2_mocks
        model = Wav2Vec2ASRModel(_make_config())
        assert model.processor == m.processor_instance

    @patch("builtins.open", create=True)
    @patch("os.makedirs")
    @patch("torch.save")
    def test_save_pretrained(
        self, _mock_save, _mock_makedirs, _mock_open, wav2vec2_mocks: Wav2Vec2Mocks
    ):
        """Test save_pretrained method."""
        m = wav2vec2_mocks
        m.model_instance.save_pretrained = Mock()
        m.processor_instance.save_pretrained = Mock()

        model = Wav2Vec2ASRModel(_make_config())
        model.save_pretrained("/tmp/test_model")

        m.model_instance.save_pretrained.assert_called_once_with("/tmp/test_model")
        m.processor_instance.save_pretrained.assert_called_once_with("/tmp/test_model")

    def test_from_pretrained_with_config(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test from_pretrained class method with explicit config."""
        m = wav2vec2_mocks
        config = _make_config()
        model = Wav2Vec2ASRModel.from_pretrained("/tmp/test_model", config)

        m.model_class.from_pretrained.assert_called_once_with("/tmp/test_model")
        m.processor_class.from_pretrained.assert_called_once_with("/tmp/test_model")

        assert isinstance(model, Wav2Vec2ASRModel)
        assert model.config.model_type == "wav2vec2"
        assert model.config.model_name_or_path == "/tmp/test_model"
        assert model.config.loss_type == config.loss_type
        assert model.config.device == config.device

    def test_from_pretrained_without_config(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test from_pretrained infers config when not provided."""
        m = wav2vec2_mocks
        model = Wav2Vec2ASRModel.from_pretrained("/tmp/test_model")

        m.model_class.from_pretrained.assert_called_once_with("/tmp/test_model")
        assert isinstance(model, Wav2Vec2ASRModel)
        assert model.config.model_type == "wav2vec2"
        assert model.config.model_name_or_path == "/tmp/test_model"

    def test_to_device(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test moving model to different devices via nn.Module.to()."""
        model = Wav2Vec2ASRModel(_make_config())
        result = model.to("cpu")
        assert result is model

    def test_train_eval_modes(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test that nn.Module.train()/eval() correctly updates self.training."""
        model = Wav2Vec2ASRModel(_make_config())

        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

    def test_different_model_sizes(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Test initialization with different Wav2Vec2 model sizes."""
        model_names = [
            "facebook/wav2vec2-base",
            "facebook/wav2vec2-large",
            "facebook/wav2vec2-large-960h",
            "facebook/wav2vec2-base-960h",
        ]
        for model_name in model_names:
            model = Wav2Vec2ASRModel(_make_config(model_name))
            assert model.config.model_name_or_path == model_name

    def test_blank_bias_init_sets_bias(self, wav2vec2_mocks: Wav2Vec2Mocks):
        """Blank bias init sets lm_head.bias[0] to specified value."""
        m = wav2vec2_mocks
        mock_lm_head = Mock()
        mock_lm_head.bias = Mock()
        mock_lm_head.bias.data = torch.zeros(32)
        m.model_instance.lm_head = mock_lm_head

        model = Wav2Vec2ASRModel(_make_config(), blank_bias_init=-3.0)

        assert model.model.lm_head.bias.data[0].item() == -3.0
        assert model.model.lm_head.bias.data[1].item() == 0.0

    def test_blank_bias_init_none_leaves_bias_unchanged(
        self, wav2vec2_mocks: Wav2Vec2Mocks
    ):
        """When blank_bias_init is None, lm_head bias is not modified."""
        m = wav2vec2_mocks
        mock_lm_head = Mock()
        mock_lm_head.bias = Mock()
        mock_lm_head.bias.data = torch.ones(32) * 0.5
        m.model_instance.lm_head = mock_lm_head

        model = Wav2Vec2ASRModel(_make_config())
        assert model.model.lm_head.bias.data[0].item() == 0.5


class TestWav2Vec2SSLOnlyModel:
    """Test Wav2Vec2ASRModel with SSL-only models that lack a tokenizer (e.g. XLS-R)."""

    def test_ssl_only_model_falls_back_to_feature_extractor(
        self, wav2vec2_ssl_mocks: Wav2Vec2Mocks
    ):
        """SSL-only models (XLS-R) lack a tokenizer; processor load should fall back."""
        m = wav2vec2_ssl_mocks

        # Processor.from_pretrained fails for SSL-only models (no tokenizer)
        m.processor_class.from_pretrained.side_effect = OSError(
            "Can't load tokenizer for 'facebook/wav2vec2-xls-r-300m'"
        )

        # Feature extractor loads fine
        with patch(f"{_MODULE}.Wav2Vec2FeatureExtractor") as mock_feat_ext:
            mock_fe_instance = Mock()
            mock_feat_ext.from_pretrained.return_value = mock_fe_instance

            # Processor constructed from feature_extractor + placeholder tokenizer
            mock_processor_instance = Mock()
            m.processor_class.return_value = mock_processor_instance

            # SSL path: loads Wav2Vec2Model then transfers backbone weights
            with patch(f"{_MODULE}.Wav2Vec2Model") as mock_ssl_model_class:
                mock_ssl_instance = Mock()
                mock_ssl_model_class.from_pretrained.return_value = mock_ssl_instance

                config = _make_config("facebook/wav2vec2-xls-r-300m")
                model = Wav2Vec2ASRModel(config)

            # Feature extractor fallback was used
            mock_feat_ext.from_pretrained.assert_called_once_with(
                "facebook/wav2vec2-xls-r-300m"
            )
            # Processor constructed with feature_extractor + placeholder tokenizer
            m.processor_class.assert_called_once()
            call_kwargs = m.processor_class.call_args[1]
            assert call_kwargs["feature_extractor"] is mock_fe_instance
            assert call_kwargs["tokenizer"] is not None
            assert model._processor == mock_processor_instance

            # SSL backbone loaded into Wav2Vec2ForCTC (not from_pretrained)
            mock_ssl_model_class.from_pretrained.assert_called_once_with(
                "facebook/wav2vec2-xls-r-300m"
            )
            m.model_class.assert_called_once_with(m.wav2vec2_config)
            m.model_class.from_pretrained.assert_not_called()
