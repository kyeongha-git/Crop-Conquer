#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pathlib import Path
import yaml
import logging

from utils.load_config import load_yaml_config
from utils.logging import setup_logging, get_logger


"""
test_load_config.py
-------------------
Unit tests for `utils.load_config.load_yaml_config`.
"""

def test_load_valid_yaml(tmp_path):
    """✅ 정상 YAML 로드 테스트"""
    config_content = """
    data_augmentor:
      data:
        input_dir: "data/original"
        output_dir: "data/output"
      split:
        train_ratio: 0.8
        valid_ratio: 0.1
        test_ratio: 0.1
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(config_content, encoding="utf-8")

    cfg = load_yaml_config(yaml_path)
    assert "data_augmentor" in cfg
    assert cfg["data_augmentor"]["split"]["train_ratio"] == 0.8


def test_file_not_found(tmp_path):
    """❌ 존재하지 않는 파일 경로"""
    nonexistent_path = tmp_path / "no_such_file.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_config(nonexistent_path)


def test_invalid_yaml_syntax(tmp_path):
    """❌ YAML 문법 오류"""
    invalid_yaml_content = """
    data_augmentor:
      data:
        input_dir: "data/original"
        output_dir: "data/output"
      split:
        train_ratio: 0.8
        valid_ratio: 0.1
        test_ratio: 0.1
        test_ratio: 0.1   # duplicated key
      invalid_block: [ unclosed_bracket
    """
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text(invalid_yaml_content, encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_yaml_config(yaml_path)


def test_invalid_yaml_structure(tmp_path):
    """❌ YAML의 최상단 구조가 dict가 아닐 경우"""
    yaml_path = tmp_path / "invalid_type.yaml"
    yaml_path.write_text("- item1\n- item2", encoding="utf-8")  # 리스트 형태

    with pytest.raises(ValueError):
        load_yaml_config(yaml_path)


def test_path_is_resolved(tmp_path):
    """📄 Path.resolve()가 절대 경로로 변환되는지 테스트"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("root: test", encoding="utf-8")

    cfg = load_yaml_config(yaml_path)

    assert cfg["root"] == "test"
    assert yaml_path.resolve().exists()


def test_stdout_message_contains_loaded_path(tmp_path, capsys):
    """🖨️ 정상 로드 시 콘솔 출력 메시지 확인"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("root: test", encoding="utf-8")

    _ = load_yaml_config(yaml_path)
    out, _ = capsys.readouterr()

    assert "[✓] Loaded configuration from:" in out
    assert str(yaml_path.resolve()) in out


"""
test_logging.py
---------------
Unit tests for `utils.logging`.
"""

def test_setup_logging_creates_log_dir_and_file(tmp_path):
    """📁 로그 디렉토리와 로그 파일이 생성되는지 테스트"""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)

    # 디렉토리 생성 확인
    assert log_dir.exists(), "Log directory not created"

    # 로그 파일 생성 확인
    log_files = list(log_dir.glob("run_*.log"))
    assert len(log_files) == 1, "Log file not created"
    assert log_files[0].suffix == ".log"


def test_setup_logging_registers_handlers(tmp_path):
    """🧩 StreamHandler와 FileHandler가 모두 등록되는지 테스트"""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)

    root_logger = logging.getLogger()
    handler_types = [type(h).__name__ for h in root_logger.handlers]

    assert "StreamHandler" in handler_types, "StreamHandler not found"
    assert "FileHandler" in handler_types, "FileHandler not found"


def test_logging_writes_to_file(tmp_path):
    """📝 로그 메시지가 파일에 실제로 기록되는지 테스트"""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    logger = get_logger("test_logger")

    logger.info("Hello, logging test!")

    log_file = next(log_dir.glob("run_*.log"))
    content = log_file.read_text(encoding="utf-8")

    assert "Hello, logging test!" in content, "Message not written to log file"
    assert "INFO" in content, "INFO level not found in log content"


def test_get_logger_returns_same_instance():
    """🔁 동일 name 호출 시 동일 Logger 인스턴스를 반환하는지 테스트"""
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_a")

    assert logger_a is logger_b, "get_logger did not return the same instance"


def test_get_logger_returns_different_instances_for_different_names():
    """⚙️ 서로 다른 name일 때 서로 다른 Logger 인스턴스인지 테스트"""
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_b")

    assert logger_a is not logger_b, "Different names returned the same logger"
    assert isinstance(logger_a, logging.Logger)
    assert isinstance(logger_b, logging.Logger)
