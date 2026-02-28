"""Evaluation script for image classification."""

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data_module import get_dataset
from src.model_module import ModelFactory
from src.trainer_module import Trainer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate image classification model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional, uses saved config if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for evaluation"
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        # Try to find config in checkpoint directory
        checkpoint_path = Path(args.checkpoint)
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            logger.info(f"Loaded config from: {config_path}")
        else:
            raise FileNotFoundError(
                f"Config not found at {config_path}. "
                "Please provide --config argument."
            )

    OmegaConf.resolve(cfg)
    logger.info("Configuration loaded")

    # Get device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading validation dataset...")
    _, val_loader = get_dataset(cfg)

    # Create model
    logger.info(f"Creating model: {cfg.model.name}")
    model_builder = ModelFactory(cfg.model.name)
    model = model_builder(cfg)

    # Create trainer (for validation method)
    trainer = Trainer(model, cfg, device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)

    # Evaluate
    logger.info("Running evaluation...")
    val_loss, val_acc = trainer.validate(val_loader)

    logger.info("=" * 50)
    logger.info(f"Validation Results:")
    logger.info(f"  Loss: {val_loss:.4f}")
    logger.info(f"  Accuracy: {val_acc:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
