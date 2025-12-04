"""
Minimal environment loader for the ContextCap backend.

This module reads a `.env` file at the project root (if present) and adds
any `KEY=VALUE` pairs that are not already defined in the environment.
"""

import os
from .paths import ROOT_DIR
from pathlib import Path

ENV_PATH = ROOT_DIR / ".env"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    text = ENV_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load on import
_load_env_file()
