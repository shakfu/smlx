"""Load an exported .mlxfn Llama step and greedy-decode N tokens. Pure mlx;
no mlx_lm at inference time -- this is the equivalent of what a C runner does.

Usage: run_exported.py <model.mlxfn> <n> <id1> [<id2> ...]
"""
import sys
import mlx.core as mx

f = mx.import_function(sys.argv[1])
n = int(sys.argv[2])
prompt = [int(x) for x in sys.argv[3:]]

# Probe layer count + cache shapes by running once with empty cache.
# We need to know L, H, D ahead of time to build the empty cache list.
# Embed everything in the function's input/output count: outputs = 1 + 2*L.
# Try with a flexible probe: assume L = (len(outputs) - 1) // 2 after first call.
# To do that, we need an initial cache size; use 16 layers (Llama 3.2 1B).
# Better: read it from the function metadata if available. Otherwise hard-code.
L, H, D = 16, 8, 64

def empty_kv():
    return [mx.zeros((1, H, 0, D), dtype=mx.bfloat16) for _ in range(2 * L)]

cache = empty_kv()
tokens = mx.array([prompt], dtype=mx.int32)
out = f(tokens, *cache)
logits = out[0]
cache = list(out[1:])
ids = [int(mx.argmax(logits[:, -1, :], axis=-1).item())]
for _ in range(n - 1):
    tokens = mx.array([[ids[-1]]], dtype=mx.int32)
    out = f(tokens, *cache)
    logits = out[0]
    cache = list(out[1:])
    ids.append(int(mx.argmax(logits[:, -1, :], axis=-1).item()))

print(" ".join(str(i) for i in ids))
