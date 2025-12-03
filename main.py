#!/usr/bin/env python3
"""
BitNet-TT: BitNet b1.58 LLM on Tenstorrent Blackhole p150a

This implementation uses TT-NN for native execution on Tenstorrent hardware.

Quick start:
    # Run mini model demo
    python main.py

    # Run full 2B model demo
    python main.py --full

    # Run interactive chat
    python main.py --chat

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
        "--chat",
        action="store_true",
        help="Run interactive chat mode with full model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick self-test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    args = parser.parse_args()

    if args.test:
        run_quick_test()
    elif args.chat:
        run_interactive_chat(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif args.full:
        from examples.demo import run_full_demo
        run_full_demo()
    else:
        from examples.demo import run_mini_demo
        run_mini_demo()


def run_interactive_chat(max_tokens: int = 256, temperature: float = 0.7) -> None:
    """Run interactive chat mode with the full BitNet model."""
    print("=" * 60)
    print("BitNet-TT Interactive Chat")
    print("=" * 60)
    print()

    from bitnet_tt.utils.device import is_ttnn_available

    if not is_ttnn_available():
        print("Error: TT-NN is not available.")
        print("Please install the Tenstorrent SDK and drivers.")
        return

    print("Loading model... (this may take a while on first run)")
    print()

    from bitnet_tt.inference.generator import TextGenerator
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.utils.device import device_context
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

    with device_context() as device:
        # Load model
        try:
            state_dict, config = load_bitnet_weights("microsoft/bitnet-b1.58-2B-4T-bf16")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Please ensure you have internet access and huggingface_hub installed.")
            return

        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        # Create generator
        generator = TextGenerator(model)

        if generator.tokenizer is None:
            print("Error: Tokenizer not available.")
            return

        print("Model loaded successfully!")
        print(f"Model: BitNet b1.58 2B-4T ({config.hidden_size}d, {config.num_layers}L)")
        print()
        print("Commands:")
        print("  /quit, /exit, /q - Exit the chat")
        print("  /clear, /reset   - Clear conversation history")
        print("  /temp <value>    - Set temperature (e.g., /temp 0.5)")
        print("  /tokens <value>  - Set max tokens (e.g., /tokens 100)")
        print("  /stats           - Toggle stats display (default: on)")
        print()
        print("-" * 60)

        # Chat loop
        conversation_history = []
        current_temp = temperature
        current_max_tokens = max_tokens
        show_stats = True

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                if cmd[0] in ["/quit", "/exit", "/q"]:
                    print("\nGoodbye!")
                    break
                elif cmd[0] in ["/clear", "/reset"]:
                    conversation_history = []
                    print("Conversation cleared.")
                    continue
                elif cmd[0] == "/temp" and len(cmd) > 1:
                    try:
                        current_temp = float(cmd[1])
                        print(f"Temperature set to {current_temp}")
                    except ValueError:
                        print("Invalid temperature value.")
                    continue
                elif cmd[0] == "/tokens" and len(cmd) > 1:
                    try:
                        current_max_tokens = int(cmd[1])
                        print(f"Max tokens set to {current_max_tokens}")
                    except ValueError:
                        print("Invalid token count.")
                    continue
                elif cmd[0] == "/stats":
                    show_stats = not show_stats
                    print(f"Stats display: {'on' if show_stats else 'off'}")
                    continue
                else:
                    print("Unknown command. Use /quit to exit.")
                    continue

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})

            # Generate response with streaming
            print("\nAssistant: ", end="", flush=True)

            try:
                assistant_response = ""
                final_stats = None

                # Stream tokens
                for new_text, stats in generator.chat_streaming(
                    conversation_history,
                    max_new_tokens=current_max_tokens,
                    temperature=current_temp,
                    use_cache=True,
                ):
                    # Print new text immediately
                    print(new_text, end="", flush=True)
                    assistant_response += new_text
                    final_stats = stats

                # Clean up the response
                if generator.tokenizer.eos_token:
                    assistant_response = assistant_response.replace(
                        generator.tokenizer.eos_token, ""
                    ).strip()

                # Print stats
                if show_stats and final_stats:
                    print(final_stats)

                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": assistant_response})

            except Exception as e:
                print(f"\nError generating response: {e}")
                import traceback
                traceback.print_exc()
                # Remove the failed user message from history
                conversation_history.pop()


def run_quick_test() -> None:
    """Run a quick self-test to verify the installation."""
    print("Running quick self-test...")
    print()

    # Test imports
    print("[1/5] Testing imports...")
    try:
        from bitnet_tt.config import BitNetMiniConfig  # noqa: F401
        from bitnet_tt.layers.attention import KVCache  # noqa: F401
        from bitnet_tt.utils.device import is_ttnn_available
        from bitnet_tt.utils.quantization import weight_quant_ternary

        print("      Imports OK")
    except ImportError as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    # Check TT-NN availability
    print("[2/5] Checking TT-NN availability...")
    if is_ttnn_available():
        print("      TT-NN is available")
    else:
        print("      TT-NN is NOT available")
        print("      Install Tenstorrent SDK to run on hardware")
        sys.exit(0)

    # Test device
    print("[3/5] Testing device access...")
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
    print("[4/5] Testing quantization...")
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

    # Test KV-Cache
    print("[5/5] Testing KV-Cache...")
    try:
        from bitnet_tt.layers.attention import KVCache

        cache = KVCache()
        assert cache.key_cache is None
        assert cache.seq_len_cached == 0
        cache.reset()
        print("      KV-Cache OK")
    except Exception as e:
        print(f"      FAILED: {e}")
        sys.exit(1)

    print()
    print("All tests passed!")
    print()
    print("Run 'python main.py' to run the mini model demo")
    print("Run 'python main.py --full' to run the full 2B model demo")
    print("Run 'python main.py --chat' for interactive chat mode")


if __name__ == "__main__":
    main()
