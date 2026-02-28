"""Base model factory and registry for image classification."""

from typing import Dict, Type, Callable, Any

import torch.nn as nn

# Model registry
MODEL_REGISTRY: Dict[str, Callable[[Any], nn.Module]] = {}


def register_model(name: str) -> Callable:
    """Decorator to register a model class.

    Args:
        name: Model name identifier

    Returns:
        Decorator function

    Example:
        @register_model("resnet18")
        class ResNet18(nn.Module):
            ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        def builder(cfg: Any) -> nn.Module:
            return cls(cfg)
        MODEL_REGISTRY[name] = builder
        return cls
    return decorator


def ModelFactory(name: str) -> Callable[[Any], nn.Module]:
    """Get model builder by name.

    Args:
        name: Model name

    Returns:
        Model builder function that takes cfg and returns nn.Module

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
