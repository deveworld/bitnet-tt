"""
Regression tests for Batch32Generator trace lifecycle guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import bitnet_tt.inference.generator_batch32 as generator_batch32_module
from bitnet_tt.inference.generator_batch32 import (
    Batch32Generator,
    choose_single_user_cache_seq_len,
    choose_trace_cache_seq_len,
)


def test_choose_trace_cache_seq_len_buckets_small_requests_together() -> None:
    assert choose_trace_cache_seq_len(77) == 128
    assert choose_trace_cache_seq_len(109) == 128
    assert choose_trace_cache_seq_len(129) == 192


def test_choose_single_user_cache_seq_len_uses_exact_tile_capacity() -> None:
    assert choose_single_user_cache_seq_len(35) == 64
    assert choose_single_user_cache_seq_len(77) == 96


class _FakeTokenizer:
    eos_token_id = 99

    def __call__(self, prompt: str, return_tensors: str = "np") -> dict:
        assert return_tensors == "np"
        return {"input_ids": np.array([[11, 12]], dtype=np.int64)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "decoded"


def _make_stub_generator(enable_trace: bool = False) -> tuple[Batch32Generator, dict]:
    generator = object.__new__(Batch32Generator)
    counters = {
        "release_trace": 0,
        "release_decode_inputs": 0,
        "clear_host_decode_cache": 0,
        "ensure_kv_caches": 0,
        "prefill": 0,
        "decode_untraced": 0,
        "capture_trace": 0,
        "execute_trace": 0,
    }

    token_sequence = iter([7, _FakeTokenizer.eos_token_id])

    generator.tokenizer = _FakeTokenizer()
    generator.enable_trace = enable_trace
    generator._trace_id = 123
    generator._trace_inputs = {"embeds": object()}
    generator._trace_output = object()
    generator._trace_capture_pos = 5
    generator._decode_inputs = None
    generator._kv_caches = []

    def release_trace() -> None:
        counters["release_trace"] += 1
        generator._trace_id = None
        generator._trace_inputs = None
        generator._trace_output = None
        generator._trace_capture_pos = None

    def release_decode_inputs() -> None:
        counters["release_decode_inputs"] += 1

    def clear_host_decode_cache() -> None:
        counters["clear_host_decode_cache"] += 1

    def ensure_kv_caches(_max_seq_len: int) -> None:
        counters["ensure_kv_caches"] += 1

    def prefill_batch32(_input_ids):
        counters["prefill"] += 1
        return object(), 2

    def sample_token(_logits, _temperature, _top_k):
        return next(token_sequence)

    def execute_decode_untraced(_token_id: int, _current_pos: int):
        counters["decode_untraced"] += 1
        return object()

    def capture_trace(_token_id: int, _current_pos: int) -> bool:
        counters["capture_trace"] += 1
        return True

    def execute_trace(_token_id: int, _current_pos: int):
        counters["execute_trace"] += 1
        return object()

    generator._release_trace = release_trace
    generator._release_decode_inputs = release_decode_inputs
    generator._clear_host_decode_cache = clear_host_decode_cache
    generator._ensure_kv_caches = ensure_kv_caches
    generator._prefill_batch32 = prefill_batch32
    generator._sample_token = sample_token
    generator._execute_decode_untraced = execute_decode_untraced
    generator._capture_trace = capture_trace
    generator._execute_trace = execute_trace

    return generator, counters


def test_generate_releases_leftover_trace_before_and_after_run(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    output = generator.generate("hello", max_new_tokens=2)

    assert output == "decoded"
    assert counters["release_trace"] == 2
    assert counters["release_decode_inputs"] == 0
    assert counters["clear_host_decode_cache"] == 0
    assert counters["prefill"] == 1
    assert counters["decode_untraced"] == 1


def test_generate_streaming_releases_leftover_trace_before_and_after_run(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    chunks = list(generator.generate_streaming("hello", max_new_tokens=2))

    assert len(chunks) == 2
    assert counters["release_trace"] == 2
    assert counters["release_decode_inputs"] == 0
    assert counters["clear_host_decode_cache"] == 0
    assert counters["prefill"] == 1
    assert counters["decode_untraced"] == 1


def test_generate_requests_tight_trace_cache_capacity(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    requested = []

    def record_requested(requested_seq_len: int, bucket: int = 64, min_seq_len: int = 128) -> int:
        requested.append((requested_seq_len, bucket, min_seq_len))
        return 128

    monkeypatch.setattr(generator_batch32_module, "choose_trace_cache_seq_len", record_requested)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    generator.generate("hello", max_new_tokens=32)

    assert requested == [(35, 32, 0)]
    assert counters["ensure_kv_caches"] == 1


def test_capture_trace_skips_warmup_for_previously_compiled_position(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    decode_calls = []

    generator.device = object()
    generator._trace_id = None
    generator._trace_inputs = None
    generator._trace_output = None
    generator._trace_capture_pos = None
    generator._warmed_trace_keys = {(8, 256)}
    generator._kv_caches = [SimpleNamespace(max_seq_len=256, seq_len_cached=0)]

    generator._allocate_decode_inputs = lambda: {
        "embeds": object(),
        "pos_tensor": object(),
        "cos": object(),
        "sin": object(),
    }
    generator._copy_decode_inputs = lambda *_args: None
    generator._set_kv_cache_length = lambda _seq_len: None
    generator._release_trace = lambda: None

    def decode_step(*_args):
        decode_calls.append("decode")
        return object()

    monkeypatch.setattr(generator, "_decode_step_batch32", decode_step)
    monkeypatch.setattr(generator_batch32_module.ttnn, "begin_trace_capture", lambda *_args, **_kwargs: 77)
    monkeypatch.setattr(generator_batch32_module.ttnn, "end_trace_capture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator_batch32_module.ttnn, "synchronize_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    captured = generator._capture_trace(token_id=7, current_pos=8)

    assert captured is True
    assert decode_calls == ["decode"]
    assert generator._trace_id == 77
    assert generator._trace_capture_pos == 8


def test_capture_trace_warms_again_when_cache_shape_changes(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    decode_calls = []

    generator.device = object()
    generator._trace_id = None
    generator._trace_inputs = None
    generator._trace_output = None
    generator._trace_capture_pos = None
    generator._warmed_trace_keys = {(8, 256)}
    generator._kv_caches = [SimpleNamespace(max_seq_len=512, seq_len_cached=0)]

    generator._allocate_decode_inputs = lambda: {
        "embeds": object(),
        "pos_tensor": object(),
        "cos": object(),
        "sin": object(),
    }
    generator._copy_decode_inputs = lambda *_args: None
    generator._set_kv_cache_length = lambda _seq_len: None
    generator._release_trace = lambda: None

    def decode_step(*_args):
        decode_calls.append("decode")
        return object()

    monkeypatch.setattr(generator, "_decode_step_batch32", decode_step)
    monkeypatch.setattr(generator_batch32_module.ttnn, "begin_trace_capture", lambda *_args, **_kwargs: 77)
    monkeypatch.setattr(generator_batch32_module.ttnn, "end_trace_capture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator_batch32_module.ttnn, "synchronize_device", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    captured = generator._capture_trace(token_id=7, current_pos=8)

    assert captured is True
    assert decode_calls == ["decode", "decode"]
    assert (8, 512) in generator._warmed_trace_keys


def test_release_trace_keeps_inputs_until_explicit_cleanup(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    released_traces = []
    deallocated = []

    generator.device = object()
    generator._trace_id = 42
    generator._trace_inputs = {
        "embeds": object(),
        "pos_tensor": object(),
    }
    generator._trace_output = object()
    generator._trace_capture_pos = 8

    monkeypatch.setattr(generator_batch32_module.ttnn, "release_trace", lambda *_args: released_traces.append(True))
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda tensor: deallocated.append(tensor))

    Batch32Generator._release_trace(generator)

    assert released_traces == [True]
    assert generator._trace_id is None
    assert generator._trace_inputs is not None
    assert deallocated == []

    Batch32Generator._release_trace_inputs(generator)

    assert len(deallocated) == 2
    assert generator._trace_inputs is None
