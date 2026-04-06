"""
Device management utilities for TT-NN.

This module provides functions for managing Tenstorrent device initialization
and cleanup.
"""

from contextlib import contextmanager
import json
import os


def _parse_bool_env_value(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _prime_ttnn_config_overrides() -> None:
    """
    Apply TTNN config overrides before importing ttnn.

    TTNN reads TTNN_CONFIG_OVERRIDES during module import, so setting
    enable_model_cache afterwards is too late for cross-process reuse.
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
        overrides["enable_model_cache"] = _parse_bool_env_value(
            os.getenv("BITNET_TT_ENABLE_MODEL_CACHE"),
            default=True,
        )

    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)


_prime_ttnn_config_overrides()

# Try to import ttnn, fall back to mock for development
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


_device = None


def _get_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_bool_env(name: str, default: bool) -> bool:
    return _parse_bool_env_value(os.getenv(name), default)


def get_device(device_id: int = 0) -> "ttnn.Device":
    """
    Get or create a TT-NN device.

    Args:
        device_id: The device ID to use (default: 0)

    Returns:
        ttnn.Device instance

    Raises:
        RuntimeError: If ttnn is not available
    """
    global _device

    if not TTNN_AVAILABLE:
        raise RuntimeError("ttnn is not available. Please install the Tenstorrent SDK.")

    if _device is None:
        if hasattr(ttnn, "CONFIG") and hasattr(ttnn.CONFIG, "enable_model_cache"):
            ttnn.CONFIG.enable_model_cache = _get_bool_env(
                "BITNET_TT_ENABLE_MODEL_CACHE", default=True
            )

        open_kwargs: dict[str, int] = {
            "device_id": device_id,
            "trace_region_size": _get_int_env("BITNET_TT_TRACE_REGION_SIZE") or 20_000_000,
        }

        l1_small_size = _get_int_env("BITNET_TT_L1_SMALL_SIZE")
        if l1_small_size is not None:
            open_kwargs["l1_small_size"] = l1_small_size

        num_command_queues = _get_int_env("BITNET_TT_NUM_COMMAND_QUEUES")
        if num_command_queues is not None:
            open_kwargs["num_command_queues"] = num_command_queues

        worker_l1_size = _get_int_env("BITNET_TT_WORKER_L1_SIZE")
        if worker_l1_size is not None:
            open_kwargs["worker_l1_size"] = worker_l1_size

        _device = ttnn.open_device(**open_kwargs)
        if _get_bool_env("BITNET_TT_ENABLE_PROGRAM_CACHE", default=True) and hasattr(
            _device, "enable_program_cache"
        ):
            _device.enable_program_cache()

    return _device


def close_device() -> None:
    """Close the current TT-NN device."""
    global _device

    if _device is not None and TTNN_AVAILABLE:
        ttnn.close_device(_device)
        _device = None


@contextmanager
def device_context(device_id: int = 0):
    """
    Context manager for device lifecycle.

    Usage:
        with device_context() as device:
            # Use device
            pass
        # Device automatically closed
    """
    device = get_device(device_id)
    try:
        yield device
    finally:
        close_device()


def is_ttnn_available() -> bool:
    """Check if ttnn is available."""
    return TTNN_AVAILABLE


def get_device_info() -> dict:
    """
    Get information about the current device.

    Returns:
        Dictionary with device information
    """
    if not TTNN_AVAILABLE:
        return {"available": False}

    device = get_device()

    return {
        "available": True,
        "device_id": device.id() if hasattr(device, "id") else 0,
        # Add more device info as needed
    }
