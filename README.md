# smlx

Experiments with [mlx-c](https://github.com/ml-explore/mlx-c): a single-file
C implementation of a Llama-3 generate loop, verified bit-exact against
`mlx_lm.generate`.

## Status

- Llama 3.2 1B, both **bf16** and **4-bit** (mlx-style affine quantization).
- KV-cached prefill + token-by-token decode.
- Llama-3 RoPE scaling (the non-uniform per-dim rescaling from `config.json`).
- Sampling: greedy / temperature / top-k / top-p / seeded RNG.
- End-to-end chat via a thin Python tokenizer wrapper.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```sh
# raw ids in/out (no tokenizer)
./build/smlx \
  models/Llama-3.2-1B-Instruct-bf16/smlx.config.txt \
  models/Llama-3.2-1B-Instruct-bf16/model.safetensors \
  20 128000

# with tokenizer (.venv must have `tokenizers` installed)
python scripts/chat.py "What is the capital of France?"
python scripts/chat.py --model models/Llama-3.2-1B-Instruct-4bit \
    --temp 0.8 --top-k 40 --top-p 0.95 --seed 1 \
    "Write a haiku about Apple Silicon."

# bench vs mlx-lm
bash scripts/bench.sh models/Llama-3.2-1B-Instruct-4bit 128
```

## Sampling

Set via env vars (the chat wrapper forwards `--temp/--top-k/--top-p/--seed`):

| var | default | meaning |
|---|---|---|
| `SMLX_TEMP` | 0   | 0 = argmax; >0 = sampling |
| `SMLX_TOP_K` | 0  | 0 = disabled |
| `SMLX_TOP_P` | 1.0 | 1.0 = disabled |
| `SMLX_SEED` | 0   | per-step key derived as `seed + offset` |

## Layout

- `src/smlx.h`, `src/smlx.c` — **libsmlx** (model load, forward, KV cache, sampler).
  Public API: `smlx_config_load`, `smlx_model_load/free`, `smlx_session_new/free`,
  `smlx_generate`. Opaque handles, no mlx-c types in the API.
- `src/main.c` — thin CLI on top of libsmlx
- `examples/hello_lib.c` — minimal example linking against libsmlx
- `thirdparty/mlx-c/` — in-tree mlx-c (built by CMake as a subdirectory)
- `scripts/chat.py` — tokenizer wrapper (streams tokens)
- `scripts/ref_argmax.py`, `bench*.{sh,py}` — reference + benchmarking

## Known limits

- One model architecture (Llama). One batch dim (B=1). No chunked prefill.
- KV cache uses `concatenate` per step — O(T) memory copy per decode token.
  Switching to preallocated `slice_update` would close most of the perf gap
  vs `mlx_lm` on long sequences.
- Per-step host sync (`mlx_array_item_int32`) prevents overlapping forward
  N+1 with sampling of token N.
- No quantized embedding lookup: 4-bit embed is dequantized once at load.
