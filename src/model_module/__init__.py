"""Model module for image classification."""

from .base_model import ModelFactory, register_model
from .resnet import ResNet18

__all__ = [
    "ModelFactory",
    "register_model",
    "ResNet18",
]
