"""
Path configuration for the ContextCap backend.

This module centralises all filesystem paths so the rest of the code does not
need to worry about where prompts, data, logs, or frontend assets live.

Layout (relative to this file):

  backend/app/config/paths.py  ← this file
  backend/app/                 ← APP_DIR
  backend/                     ← BACKEND_DIR
  . (project root)            ← ROOT_DIR
"""

from pathlib import Path

# This file is located at backend/app/config/paths.py
CONFIG_PKG_DIR = Path(__file__).resolve().parent   # .../backend/app/config
APP_DIR = CONFIG_PKG_DIR.parent                    # .../backend/app
BACKEND_DIR = APP_DIR.parent                       # .../backend
ROOT_DIR = BACKEND_DIR.parent                      # project root

# Top-level resources
PROMPTS_DIR = ROOT_DIR / "prompts"
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
FRONTEND_DIR = ROOT_DIR / "frontend"
LOGS_DIR = ROOT_DIR / "logs"

# Logs
QA_LOG_DIR = LOGS_DIR / "qa"

# Static frontend: we serve the standalone `frontend/` directory under /static
STATIC_DIR = FRONTEND_DIR
