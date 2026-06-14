from pathlib import Path
from typing import Any, Dict

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str) -> Dict[str, Any]:
    full_path = get_project_root() / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
