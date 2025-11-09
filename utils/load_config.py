#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_config.py
--------------
Centralized YAML configuration loader for all modules.
Usage:
    from utils.load_config import load_yaml_config
    cfg = load_yaml_config("src/yolo_cropper/config.yaml")
"""

import yaml
from pathlib import Path

def load_yaml_config(config_path: str | Path) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path (str | Path): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If YAML is invalid.
    """
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"[Config] File not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"[Config] YAML parsing error: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"[Config] Invalid YAML structure (expected dict): {config_path}")

    print(f"[✓] Loaded configuration from: {config_path}")
    return config
