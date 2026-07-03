"""Configuration loading with sensible defaults.

Reads a YAML file (default ``config.yaml``) and deep-merges it over the built-in
defaults, so a user config only needs to specify the keys it wants to override.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency
    yaml = None


DEFAULTS: Dict[str, Any] = {
    "hotkey": {
        "combo": "ctrl+alt",
        "mode": "push_to_talk",  # or "toggle"
    },
    "audio": {
        "sample_rate": 16000,
        "device": None,
    },
    "transcription": {
        "model": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
        "warmup": True,
    },
    "cleanup": {
        "enabled": True,
        "host": "http://localhost:11434",
        "model": "gemma3:4b",
        "min_chars": 40,
        "timeout": 20,
        "temperature": 0.2,
        "keep_alive": "10m",
        "num_ctx": 2048,
    },
    "injection": {
        "method": "paste",  # or "type"
        "restore_clipboard": True,
    },
    "overlay": {
        "enabled": True,
        "position": "bottom_center",  # or "top_center"
        "margin": 56,
    },
    "tray": {
        "enabled": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None = "config.yaml") -> Dict[str, Any]:
    """Load configuration, merging a YAML file (if present) over the defaults."""
    if not path or not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)

    if yaml is None:
        raise RuntimeError("PyYAML is required to read a config file: pip install PyYAML")

    with open(path, "r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    if not isinstance(user_config, dict):
        raise ValueError(f"Config file {path!r} must contain a YAML mapping at the top level")

    return _deep_merge(DEFAULTS, user_config)


def save_override(path: str | None, updates: Dict[str, Any]) -> bool:
    """Deep-merge ``updates`` into the user's config file on disk.

    Used by the tray menu to persist choices (e.g. the cleanup model).
    Only the user's overrides are written, never the full defaults.
    Note: YAML comments in the file are not preserved.
    """
    if not path or yaml is None:
        return False

    existing: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, dict):
            existing = loaded

    merged = _deep_merge(existing, updates)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False, allow_unicode=True)
    return True
