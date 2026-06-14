from pathlib import Path
from typing import Any, Dict

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"


def load_config() -> Dict[str, Any]:
    """
    Loads project configuration from YAML file.

    Returns:
        Dictionary containing project configuration.

    Raises:
        FileNotFoundError: If settings.yaml is not found.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def get_project_root() -> Path:
    """
    Returns project root directory.

    Returns:
        Path object pointing to project root.
    """

    return ROOT_DIR