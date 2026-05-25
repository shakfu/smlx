# smlx

Experiments with [mlx-c](https://github.com/ml-explore/mlx-c): a small C
implementation of Llama / Qwen generate loops, verified bit-exact against
`mlx_lm.generate`, with a fully self-contained text-chat binary (no Python).

## Status

- Architectures: **Llama 3.x**, **Qwen 2 / 2.5** (QKV projection bias),
  **Qwen 3** (QK-Norm). All auto-detected at load.
- Both **bf16/fp16** and **4-bit** (mlx-style affine quantization).
- KV-cached prefill + token-by-token decode (preallocated slots).
- Llama-3 RoPE scaling (the non-uniform per-dim rescaling from `config.json`).
- Sampling: greedy / temperature / top-k / top-p / seeded RNG.
- Self-contained text chat (`smlx chat`): tokenizer + chat template compiled
  in, single-shot and interactive modes, token-by-token streaming. No Python.

All of the above verified **bit-exact** against `mlx_lm.generate` on a real
multi-token prompt. (Single-token / out-of-context prompts can hit bf16
near-ties where the top-2 logits are within one ULP; argmax may differ from
the reference there without indicating a bug.)

## Build

```sh
make            # configure (first run) + build everything
```

Or drive CMake directly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Convenience targets: `make repl`, `make chat MODEL=... PROMPT=...`,
`make smoke`, `make test` (bit-exact vs `mlx_lm`), `make bench`, `make clean`,
`make distclean`, `make help`. The first build compiles mlx + the Rust
tokenizer crate, so it takes a few minutes; later builds are incremental.

## Run

One binary, two subcommands: **`smlx chat`** (self-contained text chat — no
Python; tokenizer via tokenizers-cpp + chat template via minja, both compiled
in) and **`smlx ids`** (the bare engine: token ids in, ids out).

```sh
# interactive chat (multi-turn; Ctrl-D or /exit to quit)
./build/smlx chat -m models/Qwen3-4B-MLX-4bit --no-think
make repl MODEL=models/Qwen3-4B-MLX-4bit ARGS=--no-think     # same via make

# single-shot (-p)
./build/smlx chat -m models/Qwen3-4B-MLX-4bit -p "What is the capital of France?" --no-think
make chat MODEL=models/Llama-3.2-1B-Instruct-4bit PROMPT="Write a haiku." ARGS="--temp 0.8 --seed 1"

# raw token ids in/out
./build/smlx ids \
  models/Llama-3.2-1B-Instruct-bf16/smlx.config.txt \
  models/Llama-3.2-1B-Instruct-bf16/model.safetensors \
  20 128000
```

`smlx chat` flags: `-m/--model` (required), `-p/--prompt` (single-shot; omit
for interactive), `--max` (answer-token budget, default 512), `--think-budget`
(max thinking tokens before `</think>` is forced, default 4096; 0 disables —
total decode = think-budget + max), `--temp`, `--top-k`, `--top-p`, `--seed`,
`--raw`, `--no-think` (disables Qwen3 thinking mode; ignored by other models).

`smlx ids` takes sampling/EOS via env vars (`SMLX_TEMP/TOP_K/TOP_P/SEED/EOS`).

## Sampling

`smlx chat` takes sampling as flags; `smlx ids` reads the same controls from
env vars:

| flag (`smlx chat`) | env var (`smlx ids`) | default | meaning |
|---|---|---|---|
| `--temp F`   | `SMLX_TEMP`  | 0   | 0 = argmax; >0 = sampling |
| `--top-k K`  | `SMLX_TOP_K` | 0   | 0 = disabled |
| `--top-p F`  | `SMLX_TOP_P` | 1.0 | 1.0 = disabled |
| `--seed S`   | `SMLX_SEED`  | 0   | per-step key derived as `seed + offset` |

`smlx ids` also takes `SMLX_EOS` (comma-separated stop ids); `smlx chat`
derives stop ids automatically from the model's config.

## Layout

- `src/smlx.h`, `src/smlx.c` — **libsmlx** (model load, forward, KV cache, sampler).
  Public API: `smlx_config_load`, `smlx_model_load/free`, `smlx_session_new/free`,
  `smlx_generate`. Opaque handles, no mlx-c types in the API.
- `src/main.cpp` — unified `smlx` CLI: `chat` (tokenizers-cpp + minja, no
  Python) and `ids` (raw token ids) subcommands on top of libsmlx
- `examples/hello_lib.c` — minimal example linking against libsmlx
- `examples/run_exported.c` — separate demo: runs a Python-exported `.mlxfn`
- `thirdparty/mlx-c/` — in-tree mlx-c (built by CMake as a subdirectory)
- `thirdparty/tokenizers-cpp/` — HF tokenizer (Rust, statically linked)
- `thirdparty/minja/`, `thirdparty/nlohmann/` — vendored headers for chat templates
- `scripts/ref_*.py`, `bench*.{sh,py}`, `test.sh` — reference + benchmarking + correctness

## Known limits

- One batch dim (B=1). No chunked prefill; very long prompts stress memory.
- Prefill ~half of `mlx_lm`'s speed (no `mx.compile`-equivalent fast path).
- KV cache grows unbounded — fine for a chat, would OOM a long-running server.
- Per-step host sync (`mlx_array_item_int32`) prevents overlapping forward
  N+1 with sampling of token N.
- No quantized embedding lookup: 4-bit embed is dequantized once at load.
- Architectures: Llama 3.x, Qwen 2/2.5/3 dense. No MoE, multimodal, or
  sharded (8B+) weights yet.
