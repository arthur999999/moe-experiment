"""Tests for model_loader module."""

import pytest
from pathlib import Path

from moe_experiment.model_loader import load_model, list_available_models


def test_load_model_nonexistent():
    """Test loading a non-existent model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/path/model.gguf")


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()
    assert isinstance(models, list)
