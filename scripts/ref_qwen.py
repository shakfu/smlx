"""Greedy decode with mlx-lm for an arbitrary model dir.
Usage: ref_qwen.py <model_dir> <n_tokens> <id1> [<id2> ...]
"""
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

model_dir = sys.argv[1]
n = int(sys.argv[2])
prompt_ids = [int(x) for x in sys.argv[3:]]

model, _tok = load(model_dir)
cache = make_prompt_cache(model)
sampler = make_sampler(temp=0.0)

x = mx.array(prompt_ids, dtype=mx.int32)[None, :]
out = []
for _ in range(n):
    logits = model(x, cache=cache)
    nxt = int(sampler(logits[:, -1, :]).item())
    out.append(nxt)
    x = mx.array([[nxt]], dtype=mx.int32)

print(" ".join(str(i) for i in out))
