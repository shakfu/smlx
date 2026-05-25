"""Export a Llama-3 single-step forward as an mlx function file.

The exported function signature is:

    step(tokens, *flat_cache) -> (logits, *new_flat_cache)

where flat_cache = [k0, v0, k1, v1, ..., kL-1, vL-1], each of shape
[1, n_kv_heads, T_past, head_dim]. tokens has shape [1, T_new].

The cache layout is a plain list of arrays (no KVCache helper class) using
concatenation -- simplest for tracing under shapeless=True.

Usage:
    python scripts/export_llama.py <model_dir> <out_file.mlxfn>
"""
import sys
import mlx.core as mx
from mlx_lm import load



class ConcatCache:
    """Minimal cache: pure concat along axis=2. No slot preallocation, no
    slice_update. Traces cleanly under shapeless export because every op is
    shape-polymorphic on the time axis."""
    def __init__(self, keys=None, values=None):
        self.keys = keys
        self.values = values
        self.offset = 0 if keys is None else keys.shape[2]

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = self.keys.shape[2]
        return self.keys, self.values

    # mlx-lm queries these.
    @property
    def state(self):
        return self.keys, self.values

    @property
    def meta_state(self):
        return ()

    @meta_state.setter
    def meta_state(self, v):
        pass

    def is_trimmable(self):
        return False


def main():
    model_dir = sys.argv[1]
    out_file = sys.argv[2]

    model, _tok = load(model_dir)
    cfg = model.args
    L = cfg.num_hidden_layers
    H = cfg.num_key_value_heads
    D = (cfg.head_dim if hasattr(cfg, "head_dim") and cfg.head_dim
         else cfg.hidden_size // cfg.num_attention_heads)
    print(f"layers={L} kv_heads={H} head_dim={D}")

    def step(tokens, *flat_cache):
        # If past sequence length is 0, treat as fresh (pass None into the cache).
        caches = []
        for i in range(L):
            k = flat_cache[2 * i]; v = flat_cache[2 * i + 1]
            if k.shape[2] == 0:
                caches.append(ConcatCache())
            else:
                caches.append(ConcatCache(k, v))
        logits = model(tokens, cache=caches)
        new_flat = []
        for c in caches:
            new_flat.append(c.keys)
            new_flat.append(c.values)
        return [logits, *new_flat]

    # Example inputs: prefill on a tiny 3-token prompt with empty cache.
    tokens = mx.array([[128000, 9906, 11]], dtype=mx.int32)  # <BOS> "Hello,"
    empty_kv = [mx.zeros((1, H, 0, D), dtype=mx.bfloat16) for _ in range(2 * L)]
    # Force-realize the example outputs so trace is captured.
    out = step(tokens, *empty_kv)
    mx.eval(*out)
    print(f"trace OK: logits shape {out[0].shape}, "
          f"cache0 shape {out[1].shape}")

    print(f"exporting to {out_file} (shapeless=True)...")
    mx.export_function(out_file, step, tokens, *empty_kv, shapeless=True)
    print("done.")


if __name__ == "__main__":
    main()
