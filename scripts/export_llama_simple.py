"""Simplest possible export: full prefill on the entire sequence each call.
No KV cache. Used to demonstrate the mlx_export round-trip end-to-end.

Function signature:
    full_forward(tokens) -> logits        # tokens [1, T] -> [1, T, V]

Usage: export_llama_simple.py <model_dir> <out_file.mlxfn>
"""
import sys
import mlx.core as mx
from mlx_lm import load


def main():
    model_dir, out_file = sys.argv[1], sys.argv[2]
    model, _tok = load(model_dir)

    def full_forward(tokens):
        return model(tokens)

    # Fixed shape: prefill on exactly T_FIXED tokens. The C runner must pad.
    T_FIXED = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    tokens = mx.zeros((1, T_FIXED), dtype=mx.int32)
    tokens[0, 0] = 128000  # BOS in slot 0
    out = full_forward(tokens)
    mx.eval(out)
    print(f"trace OK: T_FIXED={T_FIXED}, logits {out.shape}")

    mx.export_function(out_file, full_forward, tokens, shapeless=False)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
