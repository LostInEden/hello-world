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
    },
    "cleanup": {
        "enabled": True,
        "host": "http://localhost:11434",
        "model": "gemma3:4b",
        "min_chars": 40,
        "timeout": 20,
        "temperature": 0.2,
    },
    "injection": {
        "method": "paste",  # or "type"
        "restore_clipboard": True,
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
