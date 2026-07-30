"""Model factory for creating ASR models of different types.

This module provides a factory pattern for creating ASR model instances
of different architectures (Whisper, Wav2Vec2, MMS, etc.) using a unified interface.
"""

from typing import Any

from .base_model import BaseASRModel, BaseASRModelConfig
from .wav2vec2_model import Wav2Vec2ASRModel
from .xeus_model import XEUSASRModel


class ASRModelRegistry:
    """Registry for ASR model types.

    This class maintains a mapping of model type names to their
    implementation classes, enabling dynamic model creation.

    Examples:
        >>> registry = ASRModelRegistry()
        >>> registry.register("custom_model", CustomASRModel)
        >>> model_class = registry.get("custom_model")
    """

    def __init__(self):
        """Initialize the registry with built-in model types."""
        self._models: dict[str, type[BaseASRModel]] = {}

        # Register built-in models
        self.register("wav2vec2", Wav2Vec2ASRModel)
        self.register("xeus", XEUSASRModel)

    def register(
        self, model_type: str, model_class: type[BaseASRModel], override: bool = False
    ) -> None:
        """Register a new model type.

        Args:
            model_type: Unique identifier for the model type
            model_class: Class implementing BaseASRModel
            override: If True, allow overriding existing registrations

        Raises:
            ValueError: If model_type already registered and override=False

        Examples:
            >>> registry = ASRModelRegistry()
            >>> registry.register("conformer", ConformerASRModel)
        """
        model_type = model_type.lower()

        if model_type in self._models and not override:
            raise ValueError(
                f"Model type '{model_type}' is already registered. "
                "Use override=True to replace it."
            )

        self._models[model_type] = model_class

    def get(self, model_type: str) -> type[BaseASRModel]:
        """Get the model class for a given type.

        Args:
            model_type: Model type identifier

        Returns:
            Model class implementing BaseASRModel

        Raises:
            ValueError: If model_type is not registered

        Examples:
            >>> registry = ASRModelRegistry()
            >>> whisper_class = registry.get("whisper")
        """
        model_type = model_type.lower()

        if model_type not in self._models:
            raise ValueError(
                f"Model type '{model_type}' not registered. "
                f"Available types: {', '.join(self.list_models())}"
            )

        return self._models[model_type]

    def list_models(self) -> list[str]:
        """Get list of all registered model types.

        Returns:
            List of registered model type names

        Examples:
            >>> registry = ASRModelRegistry()
            >>> models = registry.list_models()
            >>> print(models)
            ['whisper', 'wav2vec2', 'mms']
        """
        return list(self._models.keys())

    def is_registered(self, model_type: str) -> bool:
        """Check if a model type is registered.

        Args:
            model_type: Model type identifier

        Returns:
            True if model type is registered, False otherwise

        Examples:
            >>> registry = ASRModelRegistry()
            >>> registry.is_registered("whisper")
            True
            >>> registry.is_registered("unknown")
            False
        """
        return model_type.lower() in self._models

    def unregister(self, model_type: str) -> None:
        """Unregister a model type.

        Args:
            model_type: Model type identifier to remove

        Raises:
            ValueError: If model_type is not registered

        Examples:
            >>> registry = ASRModelRegistry()
            >>> registry.unregister("whisper")
        """
        model_type = model_type.lower()

        if model_type not in self._models:
            raise ValueError(f"Model type '{model_type}' is not registered")

        del self._models[model_type]


# Global registry instance
_global_registry = ASRModelRegistry()


class ASRModelFactory:
    """Factory for creating ASR model instances.

    This class uses the ASRModelRegistry to create model instances
    based on configuration.

    Attributes:
        registry: Model registry to use (defaults to global registry)

    Examples:
        >>> factory = ASRModelFactory()
        >>> config = BaseASRModelConfig(
        ...     model_type="whisper",
        ...     model_name_or_path="openai/whisper-tiny",
        ... )
        >>> model = factory.create(config)
    """

    def __init__(self, registry: ASRModelRegistry | None = None):
        """Initialize the factory.

        Args:
            registry: Optional custom registry. If None, uses global registry.
        """
        self.registry = registry if registry is not None else _global_registry

    def create(
        self,
        config: BaseASRModelConfig,
        from_scratch: bool = False,
        **kwargs: Any,
    ) -> BaseASRModel:
        """Create a model instance from configuration.

        Args:
            config: Model configuration
            from_scratch: If True, initialize with random weights
            **kwargs: Additional model-specific parameters (e.g., blank_bias_init)

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model type is not registered

        Examples:
            >>> factory = ASRModelFactory()
            >>> config = BaseASRModelConfig(
            ...     model_type="wav2vec2",
            ...     model_name_or_path="facebook/wav2vec2-base",
            ... )
            >>> model = factory.create(config)
        """
        model_class = self.registry.get(config.model_type)
        return model_class(config, from_scratch=from_scratch, **kwargs)

    def create_from_pretrained(
        self,
        model_type: str,
        load_path: str,
        config: BaseASRModelConfig | None = None,
    ) -> BaseASRModel:
        """Create a model by loading from a checkpoint.

        Args:
            model_type: Type of model to create
            load_path: Path to model checkpoint
            config: Optional configuration override

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model type is not registered

        Examples:
            >>> factory = ASRModelFactory()
            >>> model = factory.create_from_pretrained(
            ...     "whisper",
            ...     "/path/to/checkpoint"
            ... )
        """
        model_class = self.registry.get(model_type)
        return model_class.from_pretrained(load_path, config)


# Convenience functions for common use cases


def create_asr_model(
    config: BaseASRModelConfig | dict | None = None,
    from_scratch: bool = False,
    model_type: str | None = None,
    model_name_or_path: str | None = None,
    **kwargs: Any,
) -> BaseASRModel:
    """Create an ASR model using the factory.

    This is the main convenience function for creating models.

    Args:
        config: Model configuration (BaseASRModelConfig or dict)
        from_scratch: If True, initialize with random weights
        model_type: Model type (if not in config)
        model_name_or_path: Model name/path (if not in config)
        **kwargs: Additional model-specific parameters (e.g., blank_bias_init
            for Wav2Vec2). When config is None these are passed to
            BaseASRModelConfig; when config is provided they are forwarded
            to the model constructor.

    Returns:
        Initialized model instance

    Examples:
        >>> # Using BaseASRModelConfig
        >>> config = BaseASRModelConfig(
        ...     model_type="whisper",
        ...     model_name_or_path="openai/whisper-tiny",
        ... )
        >>> model = create_asr_model(config)

        >>> # Using dict
        >>> config_dict = {
        ...     "model_type": "wav2vec2",
        ...     "model_name_or_path": "facebook/wav2vec2-base",
        ... }
        >>> model = create_asr_model(config_dict)

        >>> # Using keyword arguments
        >>> model = create_asr_model(
        ...     model_type="mms",
        ...     model_name_or_path="facebook/mms-1b-all",
        ... )
    """
    # Convert dict to BaseASRModelConfig if needed
    if isinstance(config, dict):
        config = BaseASRModelConfig(**config)
        factory = ASRModelFactory()
        return factory.create(config, from_scratch=from_scratch, **kwargs)
    elif config is None:
        # Create config from keyword arguments
        if model_type is None or model_name_or_path is None:
            raise ValueError(
                "Either 'config' or both 'model_type' and 'model_name_or_path' must be provided"
            )

        config = BaseASRModelConfig(
            model_type=model_type, model_name_or_path=model_name_or_path, **kwargs
        )
        factory = ASRModelFactory()
        return factory.create(config, from_scratch=from_scratch)

    # Config object provided — forward kwargs to model constructor
    factory = ASRModelFactory()
    return factory.create(config, from_scratch=from_scratch, **kwargs)


def get_supported_model_types() -> list[str]:
    """Get list of supported model types.

    Returns:
        List of model type names

    Examples:
        >>> types = get_supported_model_types()
        >>> print(types)
        ['whisper', 'wav2vec2', 'mms']
    """
    return _global_registry.list_models()


def register_model(model_type: str, override: bool = False):
    """Decorator to register a custom model type.

    Args:
        model_type: Unique identifier for the model type
        override: If True, allow overriding existing registrations

    Returns:
        Decorator function

    Examples:
        >>> @register_model("custom")
        ... class CustomASRModel(BaseASRModel):
        ...     # Implementation
        ...     pass
    """

    def decorator(model_class: type[BaseASRModel]) -> type[BaseASRModel]:
        _global_registry.register(model_type, model_class, override=override)
        return model_class

    return decorator
