"""Greedy decode with mlx-lm. Prints space-separated generated token ids.

Usage:
  python scripts/ref_argmax.py <n_tokens> <prompt_id> [<prompt_id> ...]
"""
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

MODEL = "models/Llama-3.2-1B-Instruct-bf16"

n = int(sys.argv[1])
prompt_ids = [int(x) for x in sys.argv[2:]]

model, _tok = load(MODEL)
cache = make_prompt_cache(model)
sampler = make_sampler(temp=0.0)  # argmax

x = mx.array(prompt_ids, dtype=mx.int32)[None, :]
out = []
for step in range(n):
    logits = model(x, cache=cache)
    next_id = int(sampler(logits[:, -1, :]).item())
    out.append(next_id)
    x = mx.array([[next_id]], dtype=mx.int32)

print(" ".join(str(i) for i in out))
