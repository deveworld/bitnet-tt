"""
Regression tests for Batch32Generator trace lifecycle guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

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

    token_sequence = iter([7, _FakeTokenizer.eos_token_id, 7, _FakeTokenizer.eos_token_id])

    generator.tokenizer = _FakeTokenizer()
    generator.enable_trace = enable_trace
    generator._trace_id = 123
    generator._trace_inputs = {"embeds": object()}
    generator._trace_output = object()
    generator._trace_capture_pos = 5
    generator._decode_inputs = None
    generator._kv_caches = []
    generator._cached_prefill_key = None
    generator._cached_prefill_seq_len = 0
    generator._cached_prefill_logits = None
    generator._cached_prefill_cache_seq_len = 0

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
        generator._kv_caches = [SimpleNamespace(max_seq_len=_max_seq_len, seq_len_cached=0)]

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
    generator._try_reuse_prefill = Batch32Generator._try_reuse_prefill.__get__(generator, Batch32Generator)
    generator._cache_prefill = Batch32Generator._cache_prefill.__get__(generator, Batch32Generator)
    generator._make_prefill_cache_key = Batch32Generator._make_prefill_cache_key.__get__(
        generator, Batch32Generator
    )

    return generator, counters


def test_generate_keeps_reusable_trace_between_runs(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    output = generator.generate("hello", max_new_tokens=2)

    assert output == "decoded"
    assert counters["release_trace"] == 0
    assert counters["release_decode_inputs"] == 0
    assert counters["clear_host_decode_cache"] == 0
    assert counters["prefill"] == 1
    assert counters["decode_untraced"] == 1


def test_generate_streaming_keeps_reusable_trace_between_runs(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    chunks = list(generator.generate_streaming("hello", max_new_tokens=2))

    assert len(chunks) == 2
    assert counters["release_trace"] == 0
    assert counters["release_decode_inputs"] == 0
    assert counters["clear_host_decode_cache"] == 0
    assert counters["prefill"] == 1
    assert counters["decode_untraced"] == 1


def test_generate_reuses_cached_prefill_for_identical_prompt(monkeypatch) -> None:
    generator, counters = _make_stub_generator(enable_trace=False)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    first = generator.generate("hello", max_new_tokens=2)
    second = generator.generate("hello", max_new_tokens=2)

    assert first == "decoded"
    assert second == "decoded"
    assert counters["prefill"] == 1
    assert counters["ensure_kv_caches"] == 2


def test_generate_extends_from_cached_prompt_prefix(monkeypatch) -> None:
    class PrefixTokenizer:
        eos_token_id = 99

        def __call__(self, prompt: str, return_tensors: str = "np") -> dict:
            mapping = {
                "base": np.array([[11, 12]], dtype=np.int64),
                "extended": np.array([[11, 12, 13]], dtype=np.int64),
            }
            return {"input_ids": mapping[prompt]}

        def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
            return "decoded"

    generator = object.__new__(Batch32Generator)
    counters = {
        "prefill": 0,
        "ensure_kv_caches": 0,
        "decode_untraced": 0,
        "release_trace": 0,
    }
    token_sequence = iter([7, 7])

    generator.tokenizer = PrefixTokenizer()
    generator.enable_trace = False
    generator._trace_id = 123
    generator._trace_inputs = None
    generator._trace_output = None
    generator._trace_capture_pos = 2
    generator._decode_inputs = None
    generator._kv_caches = []
    generator._cached_prefill_key = None
    generator._cached_prefill_seq_len = 0
    generator._cached_prefill_logits = None
    generator._cached_prefill_cache_seq_len = 0

    def release_trace() -> None:
        counters["release_trace"] += 1
        generator._trace_id = None

    def ensure_kv_caches(max_seq_len: int) -> None:
        counters["ensure_kv_caches"] += 1
        generator._kv_caches = [SimpleNamespace(max_seq_len=max_seq_len, seq_len_cached=0)]

    def prefill_batch32(_input_ids):
        counters["prefill"] += 1
        return object(), 2

    def sample_token(_logits, _temperature, _top_k):
        return next(token_sequence)

    def execute_decode_untraced(_token_id: int, _current_pos: int):
        counters["decode_untraced"] += 1
        return object()

    generator._release_trace = release_trace
    generator._ensure_kv_caches = ensure_kv_caches
    generator._prefill_batch32 = prefill_batch32
    generator._sample_token = sample_token
    generator._execute_decode_untraced = execute_decode_untraced
    generator._try_reuse_prefill = Batch32Generator._try_reuse_prefill.__get__(generator, Batch32Generator)
    generator._try_extend_prefill_from_cached_prefix = (
        Batch32Generator._try_extend_prefill_from_cached_prefix.__get__(generator, Batch32Generator)
    )
    generator._cache_prefill = Batch32Generator._cache_prefill.__get__(generator, Batch32Generator)
    generator._make_prefill_cache_key = Batch32Generator._make_prefill_cache_key.__get__(
        generator, Batch32Generator
    )
    generator._set_kv_cache_length = lambda seq_len: setattr(
        generator._kv_caches[0], "seq_len_cached", seq_len
    )

    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    first = generator.generate("base", max_new_tokens=1)
    second = generator.generate("extended", max_new_tokens=1)

    assert first == "decoded"
    assert second == "decoded"
    assert counters["prefill"] == 1
    assert counters["decode_untraced"] == 1
    assert counters["release_trace"] == 1


def test_try_reuse_prefill_allows_smaller_requests_on_existing_cache() -> None:
    generator = object.__new__(Batch32Generator)
    cached_logits = object()
    input_ids = np.array([[11, 12]], dtype=np.int64)

    generator._cached_prefill_key = (11, 12)
    generator._cached_prefill_seq_len = 2
    generator._cached_prefill_logits = cached_logits
    generator._cached_prefill_cache_seq_len = 96
    generator._kv_caches = [SimpleNamespace(max_seq_len=96, seq_len_cached=0)]
    lengths = []
    generator._set_kv_cache_length = lengths.append

    reused = Batch32Generator._try_reuse_prefill(generator, input_ids, max_seq_len=64)

    assert reused == (cached_logits, 2)
    assert lengths == [2]


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


def test_sample_token_falls_back_when_torch_wrapper_lacks_host_tensor(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    generator.device = object()
    generator._argmax_output_tensors = []
    generator._logits_host_tensor = None
    generator._logits_host_tensors = []
    generator._supports_host_tensor_to_torch = None

    logits = SimpleNamespace(spec="spec", shape=(1, 4, 3))
    last_row = SimpleNamespace(spec="last-spec")
    host_tensor = object()
    calls = []

    monkeypatch.setattr(generator_batch32_module.ttnn, "allocate_tensor_on_host", lambda *_args: host_tensor)
    monkeypatch.setattr(generator_batch32_module.ttnn, "slice", lambda *_args, **_kwargs: last_row)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    def fake_to_torch(tensor, **kwargs):
        calls.append(kwargs)
        assert tensor is last_row
        if "host_tensor" in kwargs:
            raise TypeError("unexpected keyword argument 'host_tensor'")
        return torch.tensor([[[0.1, 0.9, 0.2]]], dtype=torch.float32)

    monkeypatch.setattr(generator_batch32_module.ttnn, "to_torch", fake_to_torch)
    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([0], dtype=torch.int64))

    token = Batch32Generator._sample_token(generator, logits, temperature=1.0, top_k=2)

    assert token == 1
    assert generator._logits_host_tensor is host_tensor
    assert generator._supports_host_tensor_to_torch is False
    assert calls == [{"host_tensor": host_tensor}, {}]


def test_sample_token_reuses_host_tensor_when_supported(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    generator.device = object()
    generator._argmax_output_tensors = []
    generator._logits_host_tensor = None
    generator._logits_host_tensors = []
    generator._supports_host_tensor_to_torch = None

    logits = SimpleNamespace(spec="spec", shape=(1, 4, 3))
    last_row = SimpleNamespace(spec="last-spec")
    host_tensor = object()
    calls = []

    monkeypatch.setattr(generator_batch32_module.ttnn, "allocate_tensor_on_host", lambda *_args: host_tensor)
    monkeypatch.setattr(generator_batch32_module.ttnn, "slice", lambda *_args, **_kwargs: last_row)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)

    def fake_to_torch(tensor, **kwargs):
        calls.append(kwargs)
        assert tensor is last_row
        assert kwargs == {"host_tensor": host_tensor}
        return torch.tensor([[[0.1, 0.2, 0.8]]], dtype=torch.float32)

    monkeypatch.setattr(generator_batch32_module.ttnn, "to_torch", fake_to_torch)
    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([0], dtype=torch.int64))

    token = Batch32Generator._sample_token(generator, logits, temperature=1.0, top_k=2)

    assert token == 2
    assert generator._supports_host_tensor_to_torch is True
    assert calls == [{"host_tensor": host_tensor}]


def test_sample_token_uses_sliced_host_logits_for_greedy(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    generator.device = object()
    generator._argmax_output_tensors = []
    generator._logits_host_tensor = None
    generator._logits_host_tensors = []
    generator._supports_host_tensor_to_torch = None

    class _FakeTensor:
        def __init__(self, label: str):
            self.label = label
            self.shape = (1, 3, 128256)
            self.spec = f"{label}-spec"

        def __getitem__(self, item):
            return _FakeTensor(f"{self.label}[{item!r}]")

        def __repr__(self) -> str:
            return self.label

    sliced = _FakeTensor("last-row")
    calls = []

    monkeypatch.setattr(
        generator_batch32_module.ttnn,
        "slice",
        lambda tensor, start, end: calls.append(("slice", tensor, tuple(start), tuple(end))) or sliced,
    )
    monkeypatch.setattr(generator_batch32_module.ttnn, "allocate_tensor_on_host", lambda *_args: "host")
    monkeypatch.setattr(
        generator_batch32_module.ttnn,
        "to_torch",
        lambda tensor, **kwargs: calls.append(("to_torch", tensor, kwargs))
        or torch.tensor([[[1.0, 5.0, 2.0]]], dtype=torch.float32),
    )
    monkeypatch.setattr(
        generator_batch32_module.ttnn,
        "deallocate",
        lambda tensor: calls.append(("deallocate", tensor)),
    )

    token = Batch32Generator._sample_token(generator, _FakeTensor("logits"), temperature=1.0, top_k=None)

    assert token == 1
    assert calls[0][0] == "slice"
    assert any(call[0] == "to_torch" for call in calls)
    assert ("deallocate", sliced) in calls


def test_sample_token_reallocates_host_tensor_when_spec_changes(monkeypatch) -> None:
    generator = object.__new__(Batch32Generator)
    generator.device = object()
    generator._argmax_output_tensors = []
    old_host_tensor = SimpleNamespace(spec="old")
    generator._logits_host_tensor = old_host_tensor
    generator._logits_host_tensors = [("old", old_host_tensor)]
    generator._supports_host_tensor_to_torch = None

    logits = SimpleNamespace(spec="new", shape=(1, 4, 3))
    last_row = SimpleNamespace(spec="last-new")
    new_host_tensor = object()
    monkeypatch.setattr(generator_batch32_module.ttnn, "slice", lambda *_args, **_kwargs: last_row)
    monkeypatch.setattr(generator_batch32_module.ttnn, "deallocate", lambda _tensor: None)
    monkeypatch.setattr(
        generator_batch32_module.ttnn,
        "allocate_tensor_on_host",
        lambda spec, _device: new_host_tensor if spec == "last-new" else None,
    )
    monkeypatch.setattr(
        generator_batch32_module.ttnn,
        "to_torch",
        lambda tensor, **kwargs: torch.tensor([[[0.1, 0.7, 0.2]]], dtype=torch.float32),
    )
    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([0], dtype=torch.int64))

    token = Batch32Generator._sample_token(generator, logits, temperature=1.0, top_k=2)

    assert token == 1
    assert generator._logits_host_tensor is new_host_tensor
    assert generator._logits_host_tensors == [("old", old_host_tensor), ("last-new", new_host_tensor)]
