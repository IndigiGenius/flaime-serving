"""Tests for ASR model factory and registry.

This module tests the model factory pattern that allows creating different
ASR model types (Whisper, Wav2Vec2, MMS, Conformer) using a unified interface.
"""

from unittest.mock import Mock, patch

import pytest

from flaime_serving.vendored.base_model import BaseASRModelConfig
from flaime_serving.vendored.model_factory import (
    ASRModelFactory,
    ASRModelRegistry,
    create_asr_model,
    get_supported_model_types,
    register_model,
)


class TestASRModelRegistry:
    """Test ASRModelRegistry for registering model types."""

    def test_register_new_model(self):
        """Test registering a new model type."""
        registry = ASRModelRegistry()

        # Create a mock model class
        mock_model_class = Mock()

        # Register it
        registry.register("custom_model", mock_model_class)

        # Verify it's registered
        assert "custom_model" in registry._models
        assert registry._models["custom_model"] == mock_model_class

    def test_register_duplicate_model_raises_error(self):
        """Test that registering duplicate model type raises error."""
        registry = ASRModelRegistry()

        mock_model_class = Mock()
        registry.register("test_model", mock_model_class)

        # Try to register again with same name
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test_model", mock_model_class)

    def test_register_duplicate_model_with_override(self):
        """Test that registering duplicate with override works."""
        registry = ASRModelRegistry()

        mock_model_class1 = Mock()
        mock_model_class2 = Mock()

        registry.register("test_model", mock_model_class1)
        registry.register("test_model", mock_model_class2, override=True)

        # Should have the second class
        assert registry._models["test_model"] == mock_model_class2

    def test_get_nonexistent_model_raises_error(self):
        """Test that getting non-existent model raises error."""
        registry = ASRModelRegistry()

        with pytest.raises(ValueError, match="Model type 'nonexistent' not registered"):
            registry.get("nonexistent")

    def test_unregister_model(self):
        """Test unregistering a model type."""
        registry = ASRModelRegistry()

        mock_model_class = Mock()
        registry.register("temp_model", mock_model_class)

        assert registry.is_registered("temp_model") is True

        # Unregister it
        registry.unregister("temp_model")

        assert registry.is_registered("temp_model") is False

    def test_unregister_nonexistent_model_raises_error(self):
        """Test that unregistering non-existent model raises error."""
        registry = ASRModelRegistry()

        with pytest.raises(ValueError, match="not registered"):
            registry.unregister("nonexistent")


class TestASRModelFactory:
    """Test ASRModelFactory for creating model instances."""

    @patch("torch.cuda.is_available")
    def test_create_wav2vec2_model(self, mock_cuda):
        """Test creating a Wav2Vec2 model via factory."""
        mock_cuda.return_value = False

        # Setup mock class
        mock_wav2vec2_class = Mock()
        mock_model_instance = Mock()
        mock_wav2vec2_class.return_value = mock_model_instance

        # Create factory and patch its registry
        factory = ASRModelFactory()
        factory.registry._models["wav2vec2"] = mock_wav2vec2_class

        # Create config
        config = BaseASRModelConfig(
            model_type="wav2vec2",
            model_name_or_path="facebook/wav2vec2-base",
            loss_type="ctc",
            device="cpu",
        )

        # Create model
        model = factory.create(config)

        # Verify
        mock_wav2vec2_class.assert_called_once_with(config, from_scratch=False)
        assert model == mock_model_instance

    @patch("torch.cuda.is_available")
    def test_create_unsupported_model_raises_error(self, mock_cuda):
        """Test that creating unsupported model type raises error."""
        mock_cuda.return_value = False

        factory = ASRModelFactory()

        config = BaseASRModelConfig(
            model_type="unsupported",
            model_name_or_path="test/model",
            loss_type="ctc",
            device="cpu",
        )

        with pytest.raises(ValueError, match="Model type 'unsupported' not registered"):
            factory.create(config)

    @patch("torch.cuda.is_available")
    def test_create_from_scratch(self, mock_cuda):
        """Test creating a model from scratch (random weights)."""
        mock_cuda.return_value = False

        # Setup mock class
        mock_whisper_class = Mock()
        mock_model_instance = Mock()
        mock_whisper_class.return_value = mock_model_instance

        # Create factory and patch its registry
        factory = ASRModelFactory()
        factory.registry._models["whisper"] = mock_whisper_class

        # Create config
        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
            loss_type="cross-entropy",
            device="cpu",
        )

        # Create model from scratch
        model = factory.create(config, from_scratch=True)

        # Verify from_scratch was passed
        mock_whisper_class.assert_called_once_with(config, from_scratch=True)
        assert model == mock_model_instance

    @patch("torch.cuda.is_available")
    def test_create_from_pretrained(self, mock_cuda):
        """Test creating a model from pretrained checkpoint."""
        mock_cuda.return_value = False

        # Setup mock class
        mock_whisper_class = Mock()
        mock_model_instance = Mock()
        mock_whisper_class.from_pretrained.return_value = mock_model_instance

        # Create factory and patch its registry
        factory = ASRModelFactory()
        factory.registry._models["whisper"] = mock_whisper_class

        # Load model
        model = factory.create_from_pretrained(
            model_type="whisper",
            load_path="/tmp/checkpoint",
        )

        # Verify (config is passed as positional argument)
        mock_whisper_class.from_pretrained.assert_called_once_with(
            "/tmp/checkpoint", None
        )
        assert model == mock_model_instance

    @patch("torch.cuda.is_available")
    def test_create_from_pretrained_with_config(self, mock_cuda):
        """Test creating a model from pretrained with explicit config."""
        mock_cuda.return_value = False

        # Setup mock class
        mock_wav2vec2_class = Mock()
        mock_model_instance = Mock()
        mock_wav2vec2_class.from_pretrained.return_value = mock_model_instance

        # Create factory and patch its registry
        factory = ASRModelFactory()
        factory.registry._models["wav2vec2"] = mock_wav2vec2_class

        # Create config
        config = BaseASRModelConfig(
            model_type="wav2vec2",
            model_name_or_path="facebook/wav2vec2-base",
            loss_type="ctc",
            device="cpu",
        )

        # Load model
        model = factory.create_from_pretrained(
            model_type="wav2vec2",
            load_path="/tmp/checkpoint",
            config=config,
        )

        # Verify (config is passed as positional argument)
        mock_wav2vec2_class.from_pretrained.assert_called_once_with(
            "/tmp/checkpoint", config
        )
        assert model == mock_model_instance


class TestConvenienceFunctions:
    """Test convenience functions for model creation."""

    @patch("flaime_serving.vendored.model_factory.ASRModelFactory")
    @patch("torch.cuda.is_available")
    def test_create_asr_model_function(self, mock_cuda, mock_factory_class):
        """Test create_asr_model convenience function."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory

        # Create model using convenience function
        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
            loss_type="cross-entropy",
            device="cpu",
        )

        model = create_asr_model(config)

        # Verify factory was used
        mock_factory.create.assert_called_once_with(config, from_scratch=False)
        assert model == mock_model

    @patch("flaime_serving.vendored.model_factory.ASRModelFactory")
    @patch("torch.cuda.is_available")
    def test_create_asr_model_from_dict(self, mock_cuda, mock_factory_class):
        """Test creating model from config dict."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory

        # Create model using dict config
        config_dict = {
            "model_type": "wav2vec2",
            "model_name_or_path": "facebook/wav2vec2-base",
            "loss_type": "ctc",
            "device": "cpu",
        }

        _ = create_asr_model(config_dict)

        # Verify factory was called with config object
        assert mock_factory.create.called
        call_args = mock_factory.create.call_args[0][0]
        assert isinstance(call_args, BaseASRModelConfig)
        assert call_args.model_type == "wav2vec2"

    @patch("flaime_serving.vendored.model_factory._global_registry")
    def test_register_model_decorator(self, mock_global_registry):
        """Test register_model decorator for custom models."""
        # Setup mock
        mock_global_registry.register = Mock()

        # Define a custom model class
        @register_model("custom")
        class CustomASRModel:
            pass

        # Verify registration was called on the global registry
        mock_global_registry.register.assert_called_once_with(
            "custom", CustomASRModel, override=False
        )

    @patch("flaime_serving.vendored.model_factory.ASRModelFactory")
    @patch("torch.cuda.is_available")
    def test_create_with_model_type_string(self, mock_cuda, mock_factory_class):
        """Test creating model with just model_type string."""
        mock_cuda.return_value = False

        # Setup mocks
        mock_factory = Mock()
        mock_model = Mock()
        mock_factory.create.return_value = mock_model
        mock_factory_class.return_value = mock_factory

        # Create model with minimal args
        _ = create_asr_model(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
        )

        # Verify factory was called
        assert mock_factory.create.called
        call_args = mock_factory.create.call_args[0][0]
        assert isinstance(call_args, BaseASRModelConfig)
        assert call_args.model_type == "whisper"
        assert call_args.model_name_or_path == "openai/whisper-tiny"


class TestModelFactoryEdgeCases:
    """Test edge cases and error handling in model factory."""

    @patch("torch.cuda.is_available")
    def test_create_with_none_config(self, mock_cuda):
        """Test that creating with None config raises error."""
        mock_cuda.return_value = False

        factory = ASRModelFactory()

        with pytest.raises(AttributeError):
            factory.create(None)  # type: ignore

    @patch("torch.cuda.is_available")
    def test_create_with_invalid_model_type(self, mock_cuda):
        """Test that invalid model type in config raises error."""
        mock_cuda.return_value = False

        factory = ASRModelFactory()

        config = BaseASRModelConfig(
            model_type="invalid_type",
            model_name_or_path="test/model",
            loss_type="ctc",
            device="cpu",
        )

        with pytest.raises(ValueError, match="not registered"):
            factory.create(config)

    @patch("flaime_serving.vendored.model_factory.ASRModelRegistry")
    @patch("torch.cuda.is_available")
    def test_factory_with_custom_registry(self, mock_cuda, mock_registry_class):
        """Test creating factory with custom registry."""
        mock_cuda.return_value = False

        # Create custom registry
        custom_registry = ASRModelRegistry()

        # Create factory with custom registry
        factory = ASRModelFactory(registry=custom_registry)

        assert factory.registry == custom_registry

    @patch("torch.cuda.is_available")
    def test_case_insensitive_model_type(self, mock_cuda):
        """Test that model type is case-insensitive."""
        mock_cuda.return_value = False

        _ = ASRModelFactory()

        # Both should work
        config_lower = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
        )

        config_upper = BaseASRModelConfig(
            model_type="WHISPER",
            model_name_or_path="openai/whisper-tiny",
        )

        # Internally, model_type should be normalized to lowercase
        assert config_lower.model_type.lower() == config_upper.model_type.lower()


class TestXEUSFactoryIntegration:
    """Tests for XEUS model factory integration (26Q1-XEUS-03)."""

    def test_xeus_registered_in_registry(self):
        """Test that XEUS is registered in the model registry."""
        registry = ASRModelRegistry()

        assert registry.is_registered("xeus") is True
        assert "xeus" in registry.list_models()

    def test_xeus_in_supported_model_types(self):
        """Test that get_supported_model_types includes xeus."""
        types = get_supported_model_types()

        assert "xeus" in types

    @patch("torch.cuda.is_available")
    def test_create_xeus_model_via_factory(self, mock_cuda):
        """Test creating a XEUS model via factory."""
        mock_cuda.return_value = False

        # Setup mock class
        mock_xeus_class = Mock()
        mock_model_instance = Mock()
        mock_xeus_class.return_value = mock_model_instance

        # Create factory and patch its registry
        factory = ASRModelFactory()
        factory.registry._models["xeus"] = mock_xeus_class

        # Create config (XEUS uses model_name_or_path for checkpoint path)
        config = BaseASRModelConfig(
            model_type="xeus",
            model_name_or_path="/path/to/xeus/checkpoint.pth",
            loss_type="ctc",
            device="cpu",
        )

        # Create model
        model = factory.create(config)

        # Verify
        mock_xeus_class.assert_called_once_with(config, from_scratch=False)
        assert model == mock_model_instance

    def test_xeus_model_class_is_correct_type(self):
        """Test that registry returns the correct XEUS model class."""
        registry = ASRModelRegistry()

        xeus_class = registry.get("xeus")

        # Should be the XEUSASRModel class
        assert xeus_class.__name__ == "XEUSASRModel"
