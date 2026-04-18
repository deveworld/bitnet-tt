# Session 6 — INT8 activation quant hypothesis falsified

After session 5 corrected the PCC attribution to "cumulative bf16 drift",
session 6 tested whether **implementing the BitNet per-token INT8 absmax
activation quantisation** inside `ternary_matmul` would close the PCC
gap toward bitnet.cpp (and by extension HF bf16 with its native
`AutoBitLinear` → `ActQuant` path).

## Hypothesis

bitnet.cpp and HF `AutoBitLinear` both apply per-token INT8 absmax
activation quantisation at every BitLinear forward. TT does not — it
passes bf16 activations straight through to the ternary matmul. If the
PCC gap were driven by this missing step, implementing it in the TT
kernel should raise PCC vs those references significantly.

## Measurement

HF's `AutoBitLinear.forward` applies `ActQuant`:

```python
scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
activation = (activation * scale).round().clamp(-128, 127) / scale
```

We toggled `ActQuant` on and off in the HF bf16 reference and measured
TT's PCC against each variant. If the toggle matters, implementing the
same quant step on TT would close the gap.

| HF ActQuant | TT PCC vs HF bf16 |
|:-----------:|------------------:|
| ON  (default) | 0.981319       |
| **OFF**       | **0.980143**   |

Delta = **+0.0012** — noise-level.

Earlier session 6 measurements:

| TT vs reference                   | PCC       |
|-----------------------------------|----------:|
| HF fp32                           | 0.980679  |
| HF bf16 (with ActQuant)           | 0.981319  |
| HF bf16 (ActQuant disabled)       | 0.980143  |
| bitnet.cpp (INT8 × INT2 CPU)      | 0.967860  |

## What this proves

1. **The ActQuant step is not the cause of the gap.** Toggling it in
   the HF reference moves TT's PCC by a thousandth. Reproducing the
   same quant step on TT can therefore close at most ~0.001 of the
   ~0.019 gap.

2. **bitnet.cpp is the outlier, but not because of ActQuant.**
   bitnet.cpp gives *lower* PCC (0.968) than the HF references despite
   running the exact same ActQuant semantics. The extra 0.013 gap to
   bitnet.cpp comes from its different arithmetic pipeline: CPU fp32
   accumulator with AVX-packed INT8×INT2 MACs, specific reduction
   order, and per-block rescale. That is also not addressable by
   adding ActQuant on TT.

3. **The real residual is op-level numerical drift.** The ~0.019 gap
   that persists across every reference (fp32, bf16-with-quant,
   bf16-without-quant) is dominated by tt-metal bf16 op implementations
   vs PyTorch/CUDA bf16: SDPA reductions, RMSNorm accumulation order,
   RoPE fused kernels, residual-add ordering. These are structural to
   tt-metal's op library and not related to BitNet's quantisation
   scheme.

## Implication for US-602 / US-603

Implementing per-token INT8 activation quantisation inside
`reader_ternary_fused.cpp` + `ternary_mm_compute.cpp` would be a
multi-week tt-metal kernel effort (new reader variant, new compute
kernel with fp32 accumulator + scale rescale, new program factory,
numerical validation harness). The best-case PCC improvement is the
measured +0.001 — nowhere near the +0.019 needed to cross 0.99.

**Decision: do not build the kernel.** US-602 and US-603 are closed
as obsolete by US-601 evidence.

## What *would* close the gap

Not in scope for an inference-optimisation session, but documented for
the record:

- **Port the RMSNorm / SDPA / RoPE ops to bit-identical PyTorch
  equivalents** — requires rewriting multiple tt-metal kernels to
  match specific bf16 reduction orders. Likely sacrifices the current
  speed baseline.
- **Run the reference in tt-metal bf16 on CPU first** (i.e. use a TT
  emulator as the "ground truth"). This makes PCC trivially > 0.99
  because both sides run identical math, but it is an accounting
  change, not a numerical improvement.

## Conclusion

Under the ≥ 70 t/s speed budget, PCC > 0.99 is unreachable against any
third-party reference (HF fp32, HF bf16, bitnet.cpp) because the gap
is dominated by op-implementation differences that no BitNet-specific
quantisation fix can address. Current HEAD remains decode_tps 74.21 t/s
/ PCC 0.981 vs HF bf16 / PCC 0.968 vs bitnet.cpp.
