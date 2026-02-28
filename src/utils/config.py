"""Configuration dataclasses for type-safe config handling."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class NormalizeConfig:
    """Normalization configuration."""
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass(frozen=True)
class RandomCropConfig:
    """Random crop augmentation configuration."""
    size: int = 32
    padding: int = 4


@dataclass(frozen=True)
class RandomHorizontalFlipConfig:
    """Random horizontal flip configuration."""
    p: float = 0.5


@dataclass(frozen=True)
class TrainTransformConfig:
    """Training transform configuration."""
    random_crop: RandomCropConfig = field(default_factory=RandomCropConfig)
    random_horizontal_flip: RandomHorizontalFlipConfig = field(default_factory=RandomHorizontalFlipConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass(frozen=True)
class ValTransformConfig:
    """Validation transform configuration."""
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration."""
    name: str = "cifar10"
    num_classes: int = 10
    data_dir: str = "./data"
    image_size: int = 32
    num_channels: int = 3
    train_transform: TrainTransformConfig = field(default_factory=TrainTransformConfig)
    val_transform: ValTransformConfig = field(default_factory=ValTransformConfig)
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""
    name: str = "resnet18"
    num_classes: int = 10
    input_channels: int = 3
    pretrained: bool = False


@dataclass(frozen=True)
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""
    name: str = "cosine"
    step_size: int = 30
    gamma: float = 0.1
    T_max: int = 50


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 10
    min_delta: float = 0.001


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer configuration."""
    epochs: int = 50
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10
    save_best: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(frozen=True)
class Config:
    """Main application configuration."""
    exp_name: str = "cifar10_resnet18"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"
    log_interval: int = 100
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
