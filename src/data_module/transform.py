"""Data transforms for image classification."""

from typing import Dict, Any

from torchvision import transforms


def get_transforms(
    transform_config: Dict[str, Any],
    train: bool = True
) -> transforms.Compose:
    """Build transform pipeline from config.

    Args:
        transform_config: Transform configuration dict
        train: Whether building training transforms

    Returns:
        Composed transform pipeline
    """
    transform_list = []

    if train:
        # Training transforms
        if "random_crop" in transform_config:
            cfg = transform_config["random_crop"]
            transform_list.append(
                transforms.RandomCrop(cfg["size"], padding=cfg["padding"])
            )

        if "random_horizontal_flip" in transform_config:
            cfg = transform_config["random_horizontal_flip"]
            transform_list.append(
                transforms.RandomHorizontalFlip(p=cfg["p"])
            )
    else:
        # Validation transforms (usually just resize if needed)
        pass

    # Always add ToTensor and Normalize
    transform_list.append(transforms.ToTensor())

    if "normalize" in transform_config:
        cfg = transform_config["normalize"]
        transform_list.append(
            transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
        )

    return transforms.Compose(transform_list)
