#!/usr/bin/env python3
"""
Compare HuggingFace CPU greedy output against Batch32Generator on TT hardware.

This is intended as a short smoke test for output correctness on the actual
serving path. Keep max_new_tokens small to avoid long first-run compile times.
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--disable-trace", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bitnet_tt.inference.generator_batch32 import Batch32Generator
    from bitnet_tt.model.bitnet import create_model
    from bitnet_tt.utils.device import device_context
    from bitnet_tt.utils.weights import load_bitnet_weights, load_weights_to_model

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    hf_model.eval()
    hf_inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        hf_output = hf_model.generate(
            **hf_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )[0]
    hf_text = tokenizer.decode(hf_output, skip_special_tokens=True)

    with device_context() as device:
        state_dict, config = load_bitnet_weights(args.model_id)
        model = create_model(config, device)
        load_weights_to_model(model, state_dict)

        generator = Batch32Generator(
            model,
            tokenizer=tokenizer,
            enable_trace=not args.disable_trace,
        )
        if not args.skip_warmup:
            prompt_tokens = len(tokenizer(args.prompt, return_tensors="np")["input_ids"][0])
            generator.warmup(prompt_tokens=prompt_tokens, max_new_tokens=args.max_new_tokens)

        tt_text = generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
        )

    hf_ids = hf_output.tolist()
    prompt_ids = hf_inputs["input_ids"][0].tolist()
    tt_ids = tokenizer(tt_text, return_tensors="pt")["input_ids"][0].tolist()
    base = len(prompt_ids)

    print("PROMPT", repr(args.prompt))
    print("HF_TEXT", repr(hf_text))
    print("TT_TEXT", repr(tt_text))
    print("HF_SUFFIX_IDS", hf_ids[base:])
    print("TT_SUFFIX_IDS", tt_ids[base:])
    print("TEXT_MATCH", hf_text == tt_text)
    print("SUFFIX_MATCH", hf_ids[base:] == tt_ids[base:])


if __name__ == "__main__":
    main()
