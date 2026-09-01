"""Device selection utilities.

ProbeKit targets consumer GPUs (e.g. RTX 3050, 4-6 GB VRAM) first and falls
back to CPU. Every trainer / inference entry point should use `get_device()`
unless the caller explicitly overrides it.
"""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Resolve the compute device.

    Order: explicit override > CUDA > CPU. (MPS is intentionally not
    auto-selected because SAE/CLT training ops used here are unreliable on it.)
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return (
            f"cuda:{device.index or 0} ({props.name}, "
            f"{total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free)"
        )
    return str(device)


def autocast_context(device: torch.device, enabled: bool = True):
    """fp16 autocast on CUDA (saves VRAM on the 3050), no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
