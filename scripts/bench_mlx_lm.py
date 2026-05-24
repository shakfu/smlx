"""Greedy decode with mlx-lm, mirroring smlx's timing scheme.

Usage: bench_mlx_lm.py <model_dir> <max_new> "<id1 id2 ...>"
"""
import sys
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

model_dir = sys.argv[1]
max_new = int(sys.argv[2])
prompt_ids = [int(x) for x in sys.argv[3].split()]

model, _tok = load(model_dir)
cache = make_prompt_cache(model)
sampler = make_sampler(temp=0.0)

# Warm one tiny op to ensure backend is initialized so timing is meaningful.
mx.eval(mx.add(mx.array([0]), mx.array([0])))

x = mx.array(prompt_ids, dtype=mx.int32)[None, :]

t0 = time.monotonic()
logits = model(x, cache=cache)
next_id = sampler(logits[:, -1, :])
mx.eval(next_id)
t1 = time.monotonic()

out = [int(next_id.item())]
x = mx.array([[out[-1]]], dtype=mx.int32)
for _ in range(max_new - 1):
    logits = model(x, cache=cache)
    next_id = sampler(logits[:, -1, :])
    mx.eval(next_id)
    out.append(int(next_id.item()))
    x = mx.array([[out[-1]]], dtype=mx.int32)
t2 = time.monotonic()

n_prompt = len(prompt_ids)
decoded = max_new - 1
prefill_s = t1 - t0
decode_s = t2 - t1
print(f"[mlx-lm] prefill: {n_prompt} tok in {prefill_s:.3f}s = {n_prompt/prefill_s:.1f} tok/s | "
      f"decode: {decoded} tok in {decode_s:.3f}s = {decoded/decode_s:.1f} tok/s")
