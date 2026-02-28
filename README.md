# Image Classification Demo

A clean, modular deep learning image classification project using PyTorch and ResNet on CIFAR-10 dataset.

## Features

- **Modular Architecture**: Separate modules for data, model, and training logic
- **Hydra Configuration**: Hierarchical YAML configs for easy experimentation
- **Factory Pattern**: Easy to extend with new datasets and models
- **Best Practices**: Type hints, proper logging, checkpointing, early stopping
- **ResNet for CIFAR-10**: Modified ResNet-18 optimized for 32x32 images

## Project Structure

```
image_classification_demo/
├── conf/                      # Hydra configuration files
│   ├── config.yaml           # Main config
│   ├── data/
│   │   └── cifar10.yaml      # Dataset config
│   ├── model/
│   │   └── resnet18.yaml     # Model config
│   └── trainer/
│       └── default.yaml      # Training config
├── src/
│   ├── data_module/          # Data loading and transforms
│   ├── model_module/         # Model definitions
│   ├── trainer_module/       # Training loop
│   └── utils/                # Utilities (config, seed)
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── pyproject.toml            # Package configuration
└── README.md
```

## Installation

### Using conda (recommended on h800 server)

```bash
# Create conda environment
conda create -n imgcls python=3.10 -y
conda activate imgcls

# Install PyTorch (adjust cuda version as needed)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install hydra-core omegaconf tqdm

# Install project in editable mode
pip install -e .
```

### Using uv (alternative)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Basic training with default config
python train.py

# Training with custom epochs
python train.py trainer.epochs=100

# Training with different learning rate
python train.py trainer.lr=0.01

# Training with custom output directory
python train.py exp_name=my_experiment output_dir=./my_outputs
```

### Evaluation

```bash
# Evaluate best model
python evaluate.py --checkpoint outputs/cifar10_resnet18/2026-02-28/12-00-00/checkpoints/best_model.pt

# Evaluate specific checkpoint
python evaluate.py --checkpoint outputs/cifar10_resnet18/2026-02-28/12-00-00/checkpoints/checkpoint_epoch_50.pt
```

## Configuration

All configurations are managed through Hydra YAML files in `conf/`.

### Example: Custom Training Config

Create `conf/trainer/custom.yaml`:

```yaml
# @package _group_
epochs: 100
lr: 0.01
momentum: 0.9
weight_decay: 1e-4

lr_scheduler:
  name: cosine
  T_max: 100

early_stopping:
  enabled: true
  patience: 15
```

Run with:
```bash
python train.py trainer=custom
```

## Expected Results

With default configuration (ResNet-18, 50 epochs):

| Metric | Expected Value |
|--------|----------------|
| Training Accuracy | ~95-98% |
| Validation Accuracy | ~88-92% |

Training time: ~10-15 minutes on a single GPU (NVIDIA V100 or similar)

## Extending the Project

### Adding a New Dataset

1. Create dataset builder in `src/data_module/dataset.py`:

```python
@register_dataset("my_dataset")
def build_my_dataset(cfg: Any) -> tuple[Dataset, Dataset]:
    train_dataset = MyDataset(...)
    val_dataset = MyDataset(...)
    return train_dataset, val_dataset
```

2. Create config file `conf/data/my_dataset.yaml`

3. Run with `python train.py data=my_dataset`

### Adding a New Model

1. Create model class in `src/model_module/`:

```python
@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        ...
```

2. Create config file `conf/model/my_model.yaml`

3. Run with `python train.py model=my_model`

## Development

### Code Formatting

```bash
# Format code with black
black src/ train.py evaluate.py

# Lint with ruff
ruff check src/ train.py evaluate.py

# Type check with mypy
mypy src/
```

### Running Tests

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- CIFAR-10 dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
