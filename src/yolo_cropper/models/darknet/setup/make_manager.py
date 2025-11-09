#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_manager.py
---------------
Darknet Makefile Configuration & Build Manager (Config-driven)

- Receives config object from main controller (YOLOCropper)
- No direct YAML loading inside
- Compatible with third_party/darknet structure
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.append(str(ROOT_DIR))

from utils.logging import get_logger


class MakeManager:
    """
    Handles Darknet Makefile configuration and build process.
    Reads build_mode and modes flags from injected config.
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger("yolo_cropper.MakeManager")

        # --------------------------------------------------------
        # Parse nested config
        # --------------------------------------------------------
        self.cfg = config
        self.yolo_cropper_cfg = self.cfg.get("yolo_cropper", {})
        self.darknet_cfg = self.yolo_cropper_cfg.get("darknet", {})

        # Define paths
        self.darknet_dir = Path(self.darknet_cfg.get("darknet_dir", "third_party/darknet")).resolve()
        self.makefile = self.darknet_dir / "Makefile"

        if not self.makefile.exists():
            raise FileNotFoundError(f"❌ Makefile not found in {self.darknet_dir}")

        # Determine build mode
        self.build_mode = self.darknet_cfg.get("build_mode", "cpu").lower()
        self.mode_flags = self.darknet_cfg.get("modes", {}).get(self.build_mode)

        if not self.mode_flags:
            raise ValueError(
                f"❌ Invalid build_mode '{self.build_mode}' — must be one of: {list(self.darknet_cfg.get('modes', {}).keys())}"
            )

        self.jobs = self.mode_flags.get("MAKE_JOBS", 4)

        self.logger.info(f"Initialized MakeManager for {self.build_mode.upper()} mode in {self.darknet_dir}")

    # --------------------------------------------------------
    # 🔹 Public Methods
    # --------------------------------------------------------
    def configure(self, quiet: bool = True):
        """Update Makefile based on selected build mode."""
        self.logger.info(f"Configuring Makefile for {self.build_mode.upper()} build...")

        patch_flags = {k: v for k, v in self.mode_flags.items() if k != "MAKE_JOBS"}
        self._patch_makefile(patch_flags, quiet=quiet)

        self.logger.info(f"[✓] Makefile configured for {self.build_mode.upper()} mode")

    def rebuild(self, quiet: bool = True):
        """Run 'make clean' and 'make -j' to rebuild Darknet."""
        self.logger.info("Cleaning and rebuilding Darknet...")

        def _run(cmd):
            subprocess.run(
                cmd,
                cwd=self.darknet_dir,
                check=True,
                stdout=subprocess.DEVNULL if quiet else None,
                stderr=subprocess.DEVNULL if quiet else None,
            )

        _run(["make", "clean"])
        _run(["make", f"-j{self.jobs}"])
        self.logger.info(f"[✓] Darknet build complete ({self.build_mode.upper()} mode, jobs={self.jobs})")

    def verify_darknet(self, quiet: bool = True):
        """Verify that the Darknet executable exists and runs."""
        darknet_exec = self.darknet_dir / "darknet"
        if not darknet_exec.exists():
            raise FileNotFoundError("❌ Darknet executable not found. Did you build it?")

        subprocess.run(
            ["./darknet"],
            cwd=self.darknet_dir,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.DEVNULL if quiet else None,
            check=True,
        )
        self.logger.info("[✓] Darknet executable verified successfully")

    # --------------------------------------------------------
    # 🔹 Internal Helper
    # --------------------------------------------------------
    def _patch_makefile(self, flags: dict, quiet: bool = True):
        """Update key=value pairs in Makefile using sed."""
        for key, value in flags.items():
            try:
                subprocess.run(
                    ["sed", "-i", f"s/^{key}=.*/{key}={value}/", "Makefile"],
                    cwd=self.darknet_dir,
                    check=True,
                    stdout=subprocess.DEVNULL if quiet else None,
                    stderr=subprocess.DEVNULL if quiet else None,
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to patch {key} in Makefile: {e}")
                raise

        self.logger.info(f"[✓] Patched Makefile with flags: {flags}")
