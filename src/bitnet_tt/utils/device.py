"""
Device management utilities for TT-NN.

This module provides functions for managing Tenstorrent device initialization
and cleanup.
"""

from contextlib import contextmanager

# Try to import ttnn, fall back to mock for development
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


_device = None


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
        # trace_region_size required for Metal Trace (Trace capture/execute)
        # Error showed need for ~12.5MB, allocating 20MB to be safe
        _device = ttnn.open_device(device_id=device_id, trace_region_size=20_000_000)

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
