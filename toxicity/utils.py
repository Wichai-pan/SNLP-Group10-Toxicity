from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


def seed_everything(seed: int, *, seed_torch: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if seed_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    backend: str  # cuda|mps|cpu


def pick_device(prefer: Optional[str] = None) -> DeviceInfo:
    """
    prefer:
      - None: auto (cuda > mps > cpu)
      - "cuda"|"mps"|"cpu": force a backend (if available), else fall back
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for XLM-R training. Please use Python>=3.10 and install torch."
        ) from e

    prefer = (prefer or "").lower().strip() or None
    if prefer == "cuda" and torch.cuda.is_available():
        return DeviceInfo(device="cuda", backend="cuda")
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(device="mps", backend="mps")
    if prefer == "cpu":
        return DeviceInfo(device="cpu", backend="cpu")

    if torch.cuda.is_available():
        return DeviceInfo(device="cuda", backend="cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return DeviceInfo(device="mps", backend="mps")
    return DeviceInfo(device="cpu", backend="cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
