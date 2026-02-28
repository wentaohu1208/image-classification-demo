"""Dataset factory and registry for image classification."""

from typing import Dict, Type, Callable, Optional, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transform import get_transforms

# Dataset registry
DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    """Decorator to register a dataset builder function.

    Args:
        name: Dataset name identifier

    Returns:
        Decorator function

    Example:
        @register_dataset("cifar10")
        def build_cifar10(cfg: Any) -> Tuple[Dataset, Dataset]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        DATASET_REGISTRY[name] = func
        return func
    return decorator


def DatasetFactory(name: str) -> Callable:
    """Get dataset builder by name.

    Args:
        name: Dataset name

    Returns:
        Dataset builder function

    Raises:
        ValueError: If dataset name not found in registry
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name]


@register_dataset("cifar10")
def build_cifar10(cfg: Any) -> tuple[Dataset, Dataset]:
    """Build CIFAR-10 train and validation datasets.

    Args:
        cfg: Configuration object with data settings

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    data_cfg = cfg.data

    # Build transforms
    train_transform = get_transforms(
        data_cfg.train_transform,
        train=True
    )
    val_transform = get_transforms(
        data_cfg.val_transform,
        train=False
    )

    # Create datasets
    train_dataset = datasets.CIFAR10(
        root=data_cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.CIFAR10(
        root=data_cfg.data_dir,
        train=False,
        download=True,
        transform=val_transform
    )

    return train_dataset, val_dataset


def get_dataset(
    cfg: Any,
    dataset_type: Optional[str] = None
) -> tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders.

    Args:
        cfg: Configuration object
        dataset_type: Override dataset type (uses cfg.data.name if None)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset_name = dataset_type or cfg.data.name
    builder = DatasetFactory(dataset_name)

    train_dataset, val_dataset = builder(cfg)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle_train,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    return train_loader, val_loader
