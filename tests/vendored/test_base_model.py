"""Tests for base ASR model abstraction.

This module tests the BaseASRModel abstract class that provides a common
interface for all ASR model architectures (Whisper, Wav2Vec2, MMS, Conformer).
"""

from abc import ABC
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from flaime_serving.vendored.base_model import (
    BaseASRModel,
    BaseASRModelConfig,
)


class TestBaseASRModelConfig:
    """Test BaseASRModelConfig dataclass."""

    @patch("torch.cuda.is_available")
    def test_config_creation(self, mock_cuda_available):
        """Test creating a base configuration."""
        mock_cuda_available.return_value = True

        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
            loss_type="cross-entropy",
            device="cuda",
        )
        assert config.model_type == "whisper"
        assert config.model_name_or_path == "openai/whisper-tiny"
        assert config.loss_type == "cross-entropy"
        assert config.device == "cuda"

    @patch("torch.cuda.is_available")
    def test_config_defaults_cuda_available(self, mock_cuda_available):
        """Test default configuration values when CUDA is available."""
        mock_cuda_available.return_value = True

        config = BaseASRModelConfig(
            model_type="wav2vec2",
            model_name_or_path="facebook/wav2vec2-base",
        )
        assert config.model_type == "wav2vec2"
        assert config.model_name_or_path == "facebook/wav2vec2-base"
        assert config.loss_type == "ctc"  # Default for wav2vec2
        assert config.device == "cuda"  # Auto-detected as cuda

    @patch("torch.cuda.is_available")
    def test_config_defaults_cuda_unavailable(self, mock_cuda_available):
        """Test default configuration values when CUDA is unavailable."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(
            model_type="wav2vec2",
            model_name_or_path="facebook/wav2vec2-base",
        )
        assert config.model_type == "wav2vec2"
        assert config.model_name_or_path == "facebook/wav2vec2-base"
        assert config.loss_type == "ctc"  # Default for wav2vec2
        assert config.device == "cpu"  # Auto-detected as cpu

    @patch("torch.cuda.is_available")
    def test_config_all_model_types(self, mock_cuda_available):
        """Test configuration for all supported model types."""
        mock_cuda_available.return_value = False

        model_types = ["whisper", "wav2vec2", "mms", "conformer"]
        for model_type in model_types:
            config = BaseASRModelConfig(
                model_type=model_type,
                model_name_or_path=f"test/{model_type}-model",
            )
            assert config.model_type == model_type

    @patch("torch.cuda.is_available")
    def test_config_all_loss_types(self, mock_cuda_available):
        """Test configuration for all supported loss types."""
        mock_cuda_available.return_value = False

        loss_types = ["cross-entropy", "ctc", "rnnt"]
        for loss_type in loss_types:
            config = BaseASRModelConfig(
                model_type="whisper",
                model_name_or_path="openai/whisper-tiny",
                loss_type=loss_type,
            )
            assert config.loss_type == loss_type

    @patch("torch.cuda.is_available")
    def test_config_explicit_cpu_device(self, mock_cuda_available):
        """Test that explicit CPU device overrides CUDA availability."""
        mock_cuda_available.return_value = True  # CUDA available but user wants CPU

        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
            device="cpu",
        )
        assert config.device == "cpu"

    @patch("torch.cuda.is_available")
    def test_config_explicit_cuda_device(self, mock_cuda_available):
        """Test that explicit CUDA device is respected."""
        mock_cuda_available.return_value = (
            False  # CUDA not available but user specifies it
        )

        config = BaseASRModelConfig(
            model_type="whisper",
            model_name_or_path="openai/whisper-tiny",
            device="cuda",
        )
        assert config.device == "cuda"


class ConcreteASRModel(BaseASRModel):
    """Concrete implementation of BaseASRModel for testing."""

    def __init__(self, config: BaseASRModelConfig):
        """Initialize concrete model."""
        super().__init__()
        self.config = config
        self._processor = Mock()
        self.dummy_layer = nn.Linear(10, 10)

    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict:
        """Forward pass."""
        return {"loss": torch.tensor(1.0), "logits": torch.randn(2, 10, 100)}

    def generate(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate outputs."""
        return torch.randint(0, 100, (input_features.size(0), 20))

    def save_pretrained(self, save_path: str) -> None:
        """Save model."""
        pass

    @classmethod
    def from_pretrained(
        cls, load_path: str, config: BaseASRModelConfig | None = None
    ) -> "ConcreteASRModel":
        """Load model."""
        if config is None:
            config = BaseASRModelConfig(
                model_type="test", model_name_or_path="test/model"
            )
        return cls(config)

    @property
    def processor(self):
        """Get processor."""
        return self._processor

    def get_collate_fn(self, languages, packing_config=None):
        """Get collate function."""
        return lambda batch: batch

    def get_expert_collate_fn(self, lang_code, packing_config=None):
        """Get expert collate function."""
        return lambda batch: batch


class TestBaseASRModel:
    """Test BaseASRModel abstract class."""

    def test_is_abstract(self):
        """Test that BaseASRModel is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseASRModel()  # type: ignore

    def test_is_nn_module_subclass(self):
        """Test that BaseASRModel is a subclass of nn.Module."""
        assert issubclass(BaseASRModel, nn.Module)
        assert issubclass(BaseASRModel, ABC)

    @patch("torch.cuda.is_available")
    def test_concrete_implementation(self, mock_cuda_available):
        """Test that concrete implementations can be created."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        assert isinstance(model, BaseASRModel)
        assert isinstance(model, nn.Module)
        assert model.config == config

    @patch("torch.cuda.is_available")
    def test_forward_interface(self, mock_cuda_available):
        """Test that forward method works correctly."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        input_features = torch.randn(2, 80, 3000)
        labels = torch.randint(0, 100, (2, 20))

        outputs = model(input_features, labels)

        assert "loss" in outputs
        assert "logits" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert isinstance(outputs["logits"], torch.Tensor)

    @patch("torch.cuda.is_available")
    def test_forward_without_labels(self, mock_cuda_available):
        """Test forward pass without labels (inference mode)."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        input_features = torch.randn(2, 80, 3000)
        outputs = model(input_features)

        assert "logits" in outputs

    @patch("torch.cuda.is_available")
    def test_generate_interface(self, mock_cuda_available):
        """Test that generate method works correctly."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        input_features = torch.randn(2, 80, 3000)
        generated = model.generate(input_features)

        assert isinstance(generated, torch.Tensor)
        assert generated.shape[0] == input_features.shape[0]  # Same batch size

    @patch("torch.cuda.is_available")
    def test_generate_with_kwargs(self, mock_cuda_available):
        """Test generate method with additional keyword arguments."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        input_features = torch.randn(2, 80, 3000)
        generated = model.generate(
            input_features, max_length=50, num_beams=5, temperature=0.8
        )

        assert isinstance(generated, torch.Tensor)

    @patch("torch.cuda.is_available")
    def test_processor_property(self, mock_cuda_available):
        """Test that processor property is accessible."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        processor = model.processor

        assert processor is not None

    @patch("torch.cuda.is_available")
    def test_train_eval_modes(self, mock_cuda_available):
        """Test that train/eval modes work (inherited from nn.Module)."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        # Test training mode
        model.train()
        assert model.training is True

        # Test eval mode
        model.eval()
        assert model.training is False

    @patch("torch.cuda.is_available")
    def test_parameters_accessible(self, mock_cuda_available):
        """Test that model parameters are accessible (inherited from nn.Module)."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        params = list(model.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    @patch("torch.cuda.is_available")
    def test_to_device(self, mock_cuda_available):
        """Test that model can be moved to different devices."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(
            model_type="test", model_name_or_path="test/model", device="cpu"
        )
        model = ConcreteASRModel(config)

        # Move to CPU (should work regardless of CUDA availability)
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"

    @patch("torch.cuda.is_available")
    def test_state_dict(self, mock_cuda_available):
        """Test that state_dict works (inherited from nn.Module)."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        state = model.state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0

    @patch("torch.cuda.is_available")
    def test_load_state_dict(self, mock_cuda_available):
        """Test that load_state_dict works (inherited from nn.Module)."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model1 = ConcreteASRModel(config)
        model2 = ConcreteASRModel(config)

        # Get state from model1
        state = model1.state_dict()

        # Load into model2
        model2.load_state_dict(state)

        # Verify weights are the same
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    @patch("torch.cuda.is_available")
    def test_from_pretrained_classmethod(self, mock_cuda_available):
        """Test that from_pretrained class method works."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")

        model = ConcreteASRModel.from_pretrained("test/path", config)

        assert isinstance(model, ConcreteASRModel)
        assert isinstance(model, BaseASRModel)

    @patch("torch.cuda.is_available")
    def test_from_pretrained_without_config(self, mock_cuda_available):
        """Test from_pretrained can infer config if not provided."""
        mock_cuda_available.return_value = False

        model = ConcreteASRModel.from_pretrained("test/path")

        assert isinstance(model, ConcreteASRModel)
        assert model.config is not None

    @patch("torch.cuda.is_available")
    def test_save_pretrained_interface(self, mock_cuda_available):
        """Test that save_pretrained interface works."""
        mock_cuda_available.return_value = False

        config = BaseASRModelConfig(model_type="test", model_name_or_path="test/model")
        model = ConcreteASRModel(config)

        # Should not raise any exceptions
        model.save_pretrained("/tmp/test_model")


class TestBaseASRModelSubclassRequirements:
    """Test that subclasses must implement all abstract methods."""

    def test_missing_forward_raises_error(self):
        """Test that missing forward implementation raises TypeError."""

        class IncompleteModel1(BaseASRModel):
            def generate(self, input_features, **kwargs):
                pass

            def save_pretrained(self, save_path):
                pass

            @classmethod
            def from_pretrained(cls, load_path, config=None):
                pass

            @property
            def processor(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel1()  # type: ignore

    def test_missing_generate_raises_error(self):
        """Test that missing generate implementation raises TypeError."""

        class IncompleteModel2(BaseASRModel):
            def forward(self, input_features, labels=None):
                pass

            def save_pretrained(self, save_path):
                pass

            @classmethod
            def from_pretrained(cls, load_path, config=None):
                pass

            @property
            def processor(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel2()  # type: ignore

    def test_missing_save_pretrained_raises_error(self):
        """Test that missing save_pretrained implementation raises TypeError."""

        class IncompleteModel3(BaseASRModel):
            def forward(self, input_features, labels=None):
                pass

            def generate(self, input_features, **kwargs):
                pass

            @classmethod
            def from_pretrained(cls, load_path, config=None):
                pass

            @property
            def processor(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel3()  # type: ignore

    def test_missing_from_pretrained_raises_error(self):
        """Test that missing from_pretrained implementation raises TypeError."""

        class IncompleteModel4(BaseASRModel):
            def forward(self, input_features, labels=None):
                pass

            def generate(self, input_features, **kwargs):
                pass

            def save_pretrained(self, save_path):
                pass

            @property
            def processor(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel4()  # type: ignore

    def test_missing_processor_raises_error(self):
        """Test that missing processor property raises TypeError."""

        class IncompleteModel5(BaseASRModel):
            def forward(self, input_features, labels=None):
                pass

            def generate(self, input_features, **kwargs):
                pass

            def save_pretrained(self, save_path):
                pass

            @classmethod
            def from_pretrained(cls, load_path, config=None):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel5()  # type: ignore
