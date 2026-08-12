"""Model loading utilities for GGUF and Hugging Face models."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download


def load_model(
    model_path: Union[str, Path],
    repo_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> dict:
    """Load a model from local path or Hugging Face Hub.

    Args:
        model_path: Path to the model file or name
        repo_id: Optional Hugging Face repo ID to download from
        cache_dir: Optional cache directory for downloaded models

    Returns:
        Dictionary containing model data and metadata

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If invalid model format
    """
    model_path = Path(model_path)

    # If repo_id provided, download from HF Hub
    if repo_id is not None:
        model_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=str(model_path),
                cache_dir=cache_dir,
            )
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Placeholder for actual model loading logic
    # This will depend on the specific model format (GGUF, safetensors, etc.)
    return {
        "path": model_path,
        "format": model_path.suffix,
        "loaded": True,
    }


def list_available_models() -> list[str]:
    """List available models in the local cache.

    Returns:
        List of model identifiers
    """
    # Placeholder implementation
    return []
