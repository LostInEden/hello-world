"""Config loading and deep-merge behavior."""

import pytest

from whisperflow.config import DEFAULTS, _deep_merge, load_config


def test_defaults_when_no_file(tmp_path):
    config = load_config(str(tmp_path / "nope.yaml"))
    assert config == DEFAULTS
    assert config is not DEFAULTS  # must be a copy


def test_deep_merge_overrides_nested_keys():
    merged = _deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"x": 99}})
    assert base["a"]["x"] == 1


def test_partial_user_config_keeps_other_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("transcription:\n  model: tiny\n", encoding="utf-8")
    config = load_config(str(path))
    assert config["transcription"]["model"] == "tiny"
    assert config["transcription"]["compute_type"] == "float16"
    assert config["hotkey"]["combo"] == "ctrl+alt"


def test_empty_yaml_file_gives_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(str(path)) == DEFAULTS


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(path))
