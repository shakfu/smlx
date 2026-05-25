# smlx: experiments with mlx-c

This document is a writeup of what was built and learned while exploring two
approaches to running large language models from C on Apple Silicon via
[mlx-c](https://github.com/ml-explore/mlx-c), and a recommendation for where
the project should go next.

It is not API documentation; for that see `src/smlx.h` and `README.md`.

---

## 1. What we built

`smlx` is a small, single-architecture LLM runtime in C built directly on
mlx-c. The current scope:

| Component | Status |
|---|---|
| Llama 3.x family (1B, 3B; 8B+ pending sharded loader) | working |
| Qwen 2 / 2.5 (QKV bias) | working |
| Qwen 3 dense (QK-Norm) | working |
| bf16 / fp16 weights | working |
| 4-bit affine quantization (mlx-style) | working, auto-detected |
| Llama-3 RoPE scaling | working |
| KV-cached prefill + decode | working (preallocated slots + `slice_update`) |
| Sampling: greedy / temperature / top-k / top-p / seeded RNG | working |
| EOS early-stop | working |
| Streaming output | working (token-by-token in `smlx chat`) |
| Tokenizer integration | built in (`tokenizers-cpp` + `minja`, no Python) |
| C library API (`libsmlx`, opaque handles) | working |

The engine is `src/smlx.c` (~810 LOC) + `src/smlx.h` (~50 LOC). The unified CLI
is `src/main.cpp` (`smlx chat` / `smlx ids` subcommands). A `hello_lib.c`
example shows linking against the library directly.

## 2. Correctness verification

Both bf16 and 4-bit greedy decode produce **bit-identical** token sequences to
`mlx_lm.generate` on the same model and prompt, for every test run. That's the
only correctness gate that matters: if the logits at each step match the
reference, the forward pass is right.

Sampling is verified looser: deterministic with a fixed seed, varies sensibly
with seed and temperature, produces coherent text.

## 3. Performance

Bench on M-series, Llama-3.2-1B, 22-token prompt, 128-token decode. Both
implementations use greedy sampling.

| Model | Engine | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| bf16 | mlx-lm (Python) | 82.5 | 18.2 |
| bf16 | smlx (C) | 39.9 | **18.5** |
| 4-bit | mlx-lm (Python) | 84.3 → 325.0 | 55.7 → 58.5 |
| 4-bit | smlx (C) | 42.2 → 202.1 | 38.5 → **73.5** |

(Arrows show the effect of switching from concat-based KV cache to
preallocated `slice_update`. For 4-bit, decode roughly doubled.)

**Where smlx wins:** 4-bit decode is ~25% faster than mlx-lm. No Python
overhead per step; less GPU-CPU sync surface.

**Where smlx loses:** prefill is consistently slower (~50%). The likely cause
is that mlx-lm compiles the model with `mx.compile`, fusing many small ops
into single kernels for the prefill batch. smlx doesn't do this; every op is
dispatched individually through mlx-c. Adding compile to smlx is a separate
optimization not yet attempted.

**Net:** for decode-dominated workloads (chat, completion) on quantized models,
smlx is genuinely competitive. For prefill-heavy workloads (long-prompt
classification, document scoring), mlx-lm wins on the same hardware.

## 4. The two paths investigated

There are two distinct ways to use mlx-c to run real models:

### Path A: per-architecture C forward (`src/smlx.c`)

Write the model's forward pass explicitly in C against mlx-c ops. This is
what `smlx.c` does for Llama. The runtime knows about embed → RMSNorm → RoPE
→ GQA-SDPA → SwiGLU → ... at the source level.

- **Pros**
  - Total control over the inference loop (cache layout, sync points, sampling).
  - Easy to profile and optimize each op.
  - No Python dependency at runtime.
  - You learn exactly what the architecture is.
- **Cons**
  - Each architecture is new C code. Adding Qwen 2.5 is ~15 LOC (bias on QKV);
    Qwen 3 is another ~15 LOC (QK-Norm); Gemma 2 is larger; MoE is significant.
  - You re-implement what mlx-lm already has, but in a less-flexible language.

### Path B: export the model from Python, run from C (`examples/run_exported.c`)

In Python, trace the model's forward as a function and serialize it
(`mx.export_function`). In C, load the serialized graph
(`mlx_imported_function_new`) and call it (`mlx_imported_function_apply`).
The C side is fully architecture-agnostic.

- **Pros**
  - One C runner works for *any* exported model. No per-arch C code.
  - The model definition stays in Python (mlx-lm), where it's easiest to write.
  - Direct way to embed an mlx-trained model in a Swift, Zig, Rust, or other
    non-Python runtime.
- **Cons**
  - mlx-lm's stock model definitions are **not currently shapeless-export
    friendly** (see "What broke" below).
  - File is large (the whole model + graph is bundled, ~2.3 GB for 1B bf16).
  - Less control over the inference loop; sampling and KV management still
    have to live somewhere.

We verified Path B mechanically works end-to-end: a fixed-shape (T=32) export
of Llama-3.2-1B-bf16 loaded in C, run in a generate loop, produces bit-exact
output vs `mlx_lm.generate`. The C runner is **80 lines with zero
architecture-specific code**.

#### What broke when trying stateful Path B

Trying to export `(tokens, *past_kv) -> (logits, *new_kv)` with
`shapeless=True` hit two specific issues, both caused by Python integer values
getting frozen into the trace:

1. **Causal mask shape baked in.** mlx-lm's attention constructs the mask with
   `mx.triu`-style ops that take Python int dims. Calling the exported
   function with a different prompt length fails with `broadcast_shapes`.
2. **RoPE offset baked in.** `cache.offset` is a Python int. `mx.fast.rope`
   accepts an int offset and freezes it at trace time. Multi-step decode
   produces the correct first token, then garbage.

To fix this in mlx-lm itself would mean:
- Build the causal mask from tensor `.shape` arithmetic (not Python ints).
- Use `mx.fast.rope_dynamic` (takes offset as an `mx.array`) everywhere.
- Have the model return new cache arrays explicitly instead of mutating a
  `KVCache` object.

That's roughly a 200-LOC PR to mlx-lm. It hasn't been done as of the time of
this writing.

## 5. Architecture coverage today

For models hosted on `mlx-community` (Hugging Face):

- **Works directly with smlx.c:** Llama 2/3/3.1/3.2, CodeLlama, TinyLlama,
  Mistral 7B (no SWA enforcement), Vicuna/Hermes/Nous/Dolphin/etc.
  fine-tunes, SmolLM, OpenLlama, DeepSeek dense (Coder V1, Math, LLM), Yi 1.5,
  **Qwen 2 / 2.5** (QKV projection bias, auto-detected), **Qwen 3 dense**
  (QK-Norm, auto-detected). In both bf16 and 4-bit, single-shard.
- **Medium addition (~30 LOC):** Phi-3 family (fused QKV / fused gate-up
  loader), sharded safetensors loader (8B+ models).
- **Larger addition:** Gemma 2 (pre+post attn norm, GeGLU, logit softcap).
- **Out of scope without significant work:** MoE (Mixtral, Qwen3-MoE,
  DeepSeek-V2/V3, DBRX), multimodal (LLaVA, Qwen-VL, Phi-Vision), state-space
  (Mamba, RWKV), audio (Whisper), image generation, BERT/T5 encoder models,
  BitNet.

## 6. Recommendation

The interesting question is not "smlx.c or mlx_export?" — it is **what is
this project actually for?** Three honest framings, with my recommendation
at the end:

**(a) A reference implementation of Llama-style transformer inference.** This
is what smlx is best at. ~900 LOC of clear C, bit-exact against the reference,
faster than mlx-lm on 4-bit decode, and easy to read and modify. Adding Qwen
2.5 and Qwen 3 covers most of the dense-LLM landscape with another ~30 LOC.
Value: educational, profiling-friendly, embeddable.

**(b) A general-purpose mlx-c-backed inference runtime.** Path B is the right
shape for this, but it's blocked on mlx-lm. Without shapeless-export-friendly
model definitions, you can either (i) wait for upstream, (ii) maintain a fork
of mlx-lm with the rewrites, or (iii) write a single shapeless-friendly Llama
forward in Python and use it instead of mlx-lm's. Option (iii) is plausible
(~200 LOC) and would let one C runner handle every dense Llama-shaped model,
but is a real investment.

**(c) Both, with different responsibilities.** Keep smlx.c as the
"production" runtime for the architectures it supports (Llama family + Qwen
+ minor variants). Treat Path B as the "future portability" path: pursue it
when there's a concrete need to run a model architecture that smlx.c doesn't
have (e.g. MoE, multimodal) and the cost of writing it in C is higher than
the cost of writing one shapeless-friendly Python forward.

**Recommendation:** **(c).** Concretely, in this order:

1. ~~Add Qwen 2.5 and Qwen 3 dense support to smlx.c~~ **Done.** Combined
   ~40 LOC (QKV projection bias + QK-Norm, both auto-detected). Bit-exact vs
   `mlx_lm.generate` on real prompts.
2. **Add the sharded safetensors loader** (~20 LOC). Unblocks 8B+ models —
   currently the biggest gap in real-world coverage.
3. **Try `mx.compile` on the smlx-equivalent forward** in Python to confirm
   that prefill speed comes from compilation. If yes, file an mlx-c issue
   asking for an equivalent C-side fast path. (Until that exists, smlx's
   prefill will stay ~half of mlx-lm's.)
4. **Defer everything else.** Don't add Gemma, MoE, multimodal, Phi until
   there's a concrete user. They're each large enough that they justify
   either (i) waiting for export-friendly mlx-lm and going Path B, or
   (ii) being separate sub-projects.

This treats smlx as what it actually is — a fast, focused, well-understood
implementation of the dense-Llama family — rather than an attempt to be a
universal inference engine, which it can't realistically become through Path A
alone.

## 7. Loose ends and known limits

In rough order of importance:

- **Prefill is ~half mlx-lm's speed.** Almost certainly `mx.compile` related;
  not yet investigated in mlx-c.
- **KV cache grows unbounded.** Fine for a single chat; will OOM on a
  long-running streaming server. Need a max-context truncation / rolling
  window for serving.
- **Per-step host sync.** `sample_id` calls `mlx_array_item_int32` on the
  sampled id every token to control the C loop, which forces a GPU→CPU sync
  before the next forward can dispatch. mlx-lm has the same pattern but with
  different overhead. A speculative-dispatch overlap (queue N+1 before reading
  N) is possible but adds complexity.
- **No batching (B=1 only).** No chunked prefill either; very long prompts
  stress memory.
- **Sampling is correct but unoptimized.** Top-p does two argsorts and a
  take-along; could be fused. Negligible cost vs the forward pass.

(Resolved: text I/O is no longer Python-only — `smlx chat` links
`tokenizers-cpp` + `minja` and is fully self-contained, with single-shot
(`-p`) and interactive modes. The only cost is a Rust toolchain at *build*
time.)

## 8. Things explicitly chosen not to build

These came up and were deliberately skipped:

- **Speculative decoding.** Real performance win but doubles the
  implementation complexity and needs a draft model.
- **Multi-architecture C code (Gemma, Phi, etc.).** Better invested in either
  Qwen (path A) or the export-friendly Python forward (path B).
- **GGUF / GPTQ / AWQ quantization formats.** mlx 4-bit/8-bit is the native
  fit; the others would require separate dequant + matmul paths.

(Previously deferred, now done: a self-contained text binary with embedded
tokenizer + chat-template rendering — see `smlx chat`. The Rust build-time
dependency turned out to be acceptable for a fully Python-free runtime.)

---

*Last updated after task 13 (mlx_export demo).*
