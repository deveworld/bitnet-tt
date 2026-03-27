"""
Helpers for priming TTNN configuration before the first ttnn import.
"""

from __future__ import annotations

import json
import os


def parse_bool_env_value(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def prime_ttnn_config_overrides() -> None:
    """
    Apply TTNN_CONFIG_OVERRIDES before any module imports ttnn.
    """
    raw_overrides = os.environ.get("TTNN_CONFIG_OVERRIDES")
    overrides: dict[str, object]
    if raw_overrides:
        try:
            overrides = json.loads(raw_overrides)
        except json.JSONDecodeError:
            return
        if not isinstance(overrides, dict):
            return
    else:
        overrides = {}

    if "enable_model_cache" not in overrides:
        overrides["enable_model_cache"] = parse_bool_env_value(
            os.getenv("BITNET_TT_ENABLE_MODEL_CACHE"),
            default=True,
        )

    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)
