#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
logging.py
----------
Unified logging setup for all modules.
Usage:
    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message")
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.getLogger().info(f"[✓] Logging initialized. Log file: {log_file}")



def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name (str): Typically __name__.

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)
