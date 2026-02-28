"""Training script for image classification."""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_module import get_dataset
from src.model_module import ModelFactory
from src.trainer_module import Trainer
from src.utils import set_seed

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )


def get_device(device_config: str) -> torch.device:
    """Get device from config string.

    Args:
        device_config: Device configuration (auto, cpu, cuda)

    Returns:
        torch.device object
    """
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object
    """
    # Resolve config
    OmegaConf.resolve(cfg)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(str(output_dir))

    # Log configuration
    logger.info("Configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Set random seed
    set_seed(cfg.seed)
    logger.info(f"Random seed set to: {cfg.seed}")

    # Get device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader = get_dataset(cfg)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    logger.info(f"Creating model: {cfg.model.name}")
    model_builder = ModelFactory(cfg.model.name)
    model = model_builder(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, cfg, device)

    # Train
    logger.info("Starting training...")
    trainer.train(train_loader, val_loader)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
