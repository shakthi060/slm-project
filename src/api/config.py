from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "api_config.yaml"


def load_settings(config_path: Path | str | None = None) -> dict:
    path = Path(
        config_path
        or os.environ.get("SHOPEASE_API_CONFIG")
        or DEFAULT_CONFIG_PATH
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"API config must be a mapping: {path}")
    return data
