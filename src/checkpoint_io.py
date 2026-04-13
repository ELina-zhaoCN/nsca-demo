"""
Load trained CognitiveAgent weights from disk (e.g. Google Drive download folder).

Checkpoints from training (`scripts/train_all_layers.py`) are raw `state_dict` files.
World-model-only files may use `{"model_state_dict": ...}`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.cognitive_agent import CognitiveAgent, create_cognitive_agent


def _normalize_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unrecognized checkpoint format; expected state_dict or dict with model_state_dict.")


def load_cognitive_agent(
    checkpoint_path: Optional[Union[str, Path]] = None,
    *,
    use_llm: bool = False,
    map_location: str = "cpu",
) -> CognitiveAgent:
    """
    Build a CognitiveAgent and optionally load weights.

    If `checkpoint_path` is missing or the file does not exist, returns a randomly
    initialized agent (useful for UI smoke tests without weights).
    """
    agent = create_cognitive_agent(use_llm=use_llm)
    if not checkpoint_path:
        return agent
    path = Path(checkpoint_path)
    if not path.is_file():
        return agent
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    try:
        sd = _normalize_state_dict(ckpt)
    except ValueError:
        sd = ckpt if isinstance(ckpt, dict) else {}
    agent.load_state_dict(sd, strict=False)
    agent.eval()
    return agent
