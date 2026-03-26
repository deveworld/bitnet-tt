"""
Regression tests for TTNN config override priming.
"""

from __future__ import annotations

import json
import os

from bitnet_tt.utils.device import _parse_bool_env_value, _prime_ttnn_config_overrides


def test_parse_bool_env_value_handles_common_spellings() -> None:
    assert _parse_bool_env_value("1", default=False) is True
    assert _parse_bool_env_value("off", default=True) is False
    assert _parse_bool_env_value(None, default=True) is True


def test_prime_ttnn_config_overrides_adds_model_cache(monkeypatch) -> None:
    monkeypatch.delenv("TTNN_CONFIG_OVERRIDES", raising=False)
    monkeypatch.setenv("BITNET_TT_ENABLE_MODEL_CACHE", "true")

    _prime_ttnn_config_overrides()

    overrides = json.loads(os.environ["TTNN_CONFIG_OVERRIDES"])
    assert overrides["enable_model_cache"] is True


def test_prime_ttnn_config_overrides_preserves_existing_keys(monkeypatch) -> None:
    monkeypatch.setenv("TTNN_CONFIG_OVERRIDES", json.dumps({"enable_logging": True}))
    monkeypatch.setenv("BITNET_TT_ENABLE_MODEL_CACHE", "false")

    _prime_ttnn_config_overrides()

    overrides = json.loads(os.environ["TTNN_CONFIG_OVERRIDES"])
    assert overrides["enable_logging"] is True
    assert overrides["enable_model_cache"] is False
