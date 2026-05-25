"""Greedy generate N tokens by re-running the full forward at a fixed length
each step (no cache). Pads the running sequence to T_FIXED with token 0; reads
logits at the last filled position.

Usage: run_exported_simple.py <model.mlxfn> <T_FIXED> <n> <id1> [<id2> ...]
"""
import sys
import mlx.core as mx

f = mx.import_function(sys.argv[1])
T = int(sys.argv[2])
n = int(sys.argv[3])
prompt = [int(x) for x in sys.argv[4:]]

ids = list(prompt)
out_only = []
for _ in range(n):
    pos = len(ids)
    if pos > T:
        print(f"sequence exceeded T_FIXED={T}", file=sys.stderr); break
    padded = ids + [0] * (T - pos)
    tokens = mx.array([padded], dtype=mx.int32)
    logits = f(tokens)[0]
    nxt = int(mx.argmax(logits[:, pos - 1, :], axis=-1).item())
    ids.append(nxt)
    out_only.append(nxt)

print(" ".join(str(i) for i in out_only))
