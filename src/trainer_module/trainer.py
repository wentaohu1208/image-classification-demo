"""Trainer class for image classification model training."""

import logging
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for image classification models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        device: torch.device
    ) -> None:
        """Initialize trainer.

        Args:
            model: Neural network model
            cfg: Configuration object
            device: Device to train on
        """
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.trainer_cfg = cfg.trainer

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=self.trainer_cfg.lr,
            momentum=self.trainer_cfg.momentum,
            weight_decay=self.trainer_cfg.weight_decay
        )

        # Learning rate scheduler
        scheduler_cfg = self.trainer_cfg.lr_scheduler
        if scheduler_cfg.name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_cfg.T_max
            )
        elif scheduler_cfg.name == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_cfg.step_size,
                gamma=scheduler_cfg.gamma
            )
        else:
            self.scheduler = None

        # Checkpointing
        self.best_acc = 0.0
        self.checkpoint_dir = self.trainer_cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Early stopping
        self.early_stopping = self.trainer_cfg.early_stopping
        if self.early_stopping.enabled:
            self.patience_counter = 0
            self.best_loss = float('inf')

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Logging
            if batch_idx % self.cfg.log_interval == 0:
                logger.info(
                    f"Train Batch: {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.*correct/total:.2f}%"
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
        }, filepath)

        logger.info(f"Checkpoint saved: {filepath}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_acc': self.best_acc,
            }, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, filepath: str) -> int:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_acc = checkpoint.get('best_acc', 0.0)
        epoch = checkpoint.get('epoch', 0)

        logger.info(f"Checkpoint loaded: {filepath} (epoch {epoch})")
        return epoch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        num_epochs = self.trainer_cfg.epochs

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            logger.info(
                f"Train Epoch {epoch}: Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%"
            )

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            logger.info(
                f"Val Epoch {epoch}: Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            )

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")

            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc

            if epoch % self.trainer_cfg.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # Early stopping check
            if self.early_stopping.enabled:
                if val_loss < self.best_loss - self.early_stopping.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping.patience:
                        logger.info(
                            f"Early stopping triggered after {epoch} epochs"
                        )
                        break

        logger.info(f"Training completed. Best accuracy: {self.best_acc:.2f}%")
