#!/usr/bin/env python3
"""
BitNet-TT: BitNet b1.58 LLM on Tenstorrent Blackhole p150a

This implementation uses TT-NN for native execution on Tenstorrent hardware.

Quick start:
    # Run mini model demo
    python main.py

    # Run full 2B model demo
    python main.py --full

    # Run quick test
    python main.py --test
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BitNet-TT: BitNet b1.58 on Tenstorrent p150a"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run with full 2B model from HuggingFace",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick self-test",
    )
    args = parser.parse_args()

    if args.test:
        run_quick_test()
    elif args.full:
        from examples.demo import run_full_demo
        run_full_demo()
    else:
        from examples.demo import run_mini_demo
        run_mini_demo()


def run_quick_test() -> None:
    """Run a quick self-test to verify the installation."""
    print("Running quick self-test...")
    print()

    # Test imports
    print("[1/4] Testing imports...")
    try:
        from bitnet_tt.config import BitNetMiniConfig  # noqa: F401
        from bitnet_tt.utils.device import is_ttnn_available
        from bitnet_tt.utils.quantization import weight_quant_ternary

        print("      Imports OK")
    except ImportError as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    # Check TT-NN availability
    print("[2/4] Checking TT-NN availability...")
    if is_ttnn_available():
        print("      TT-NN is available")
    else:
        print("      TT-NN is NOT available")
        print("      Install Tenstorrent SDK to run on hardware")
        sys.exit(0)

    # Test device
    print("[3/4] Testing device access...")
    try:
        from bitnet_tt.utils.device import close_device, get_device

        _device = get_device()  # noqa: F841
        print("      Device opened successfully")
        close_device()
        print("      Device closed successfully")
    except Exception as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    # Test quantization
    print("[4/4] Testing quantization...")
    try:
        import numpy as np

        weight = np.random.randn(64, 128).astype(np.float32)
        weight_quant, scale = weight_quant_ternary(weight)
        unique = np.unique(weight_quant)
        print(f"      Quantized weights: {unique.tolist()}")
        print(f"      Scale factor: {scale:.4f}")
    except Exception as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    print()
    print("All tests passed!")
    print()
    print("Run 'python main.py' to run the mini model demo")
    print("Run 'python main.py --full' to run the full 2B model demo")


if __name__ == "__main__":
    main()
