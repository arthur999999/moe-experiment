# MoE Experiment

A Mixture of Experts (MoE) experiment project using Hugging Face tools and GGUF models.

## Features

- Load and work with GGUF models
- Hugging Face Hub integration
- Model downloading and management utilities

## Installation

### Using the existing virtual environment

```bash
source .venv/bin/activate
pip install -e .
```

### Fresh installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from moe_experiment import load_model

# Your code here
```

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT
