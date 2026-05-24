"""End-to-end chat wrapper around the smlx C binary.

Applies the Llama-3 chat template, tokenizes, pipes ids into ./build/smlx,
decodes ids on the way out.

Usage:
  python scripts/chat.py [--max 64] [--raw] "your prompt"

Flags:
  --max N    max new tokens (default 64)
  --raw      do NOT apply chat template; tokenize prompt as-is (single segment)
  --model D  model directory (default models/Llama-3.2-1B-Instruct-bf16)
  --bin P    path to smlx binary (default ./build/smlx)
  --config P override config.txt (default <model>/smlx.config.txt)
  --weights P override weights file (default <model>/model.safetensors)
"""
import argparse
import os
import subprocess
import sys
from tokenizers import Tokenizer

EOT_IDS_LLAMA3 = {128001, 128008, 128009}  # eos / eot variants from config.json


def llama3_chat_prompt(user_msg: str) -> str:
    # Matches tokenizer_config.json chat_template for Llama 3.2 Instruct.
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--model", default="models/Llama-3.2-1B-Instruct-bf16")
    ap.add_argument("--bin", default="./build/smlx")
    ap.add_argument("--config")
    ap.add_argument("--weights")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0, dest="top_k")
    ap.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = args.config or os.path.join(args.model, "smlx.config.txt")
    weights = args.weights or os.path.join(args.model, "model.safetensors")
    tok = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))

    prompt_text = " ".join(args.prompt)
    if not args.raw:
        prompt_text = llama3_chat_prompt(prompt_text)

    # add_special_tokens=False because we put <|begin_of_text|> in the template explicitly.
    ids = tok.encode(prompt_text, add_special_tokens=False).ids

    print(f"[prompt: {len(ids)} tokens, max_new={args.max}]", file=sys.stderr)

    env = {
        **os.environ,
        "SMLX_TEMP":  str(args.temp),
        "SMLX_TOP_K": str(args.top_k),
        "SMLX_TOP_P": str(args.top_p),
        "SMLX_SEED":  str(args.seed),
        "SMLX_EOS":   ",".join(str(i) for i in EOT_IDS_LLAMA3),
    }
    proc = subprocess.Popen(
        [args.bin, cfg, weights, str(args.max)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True, env=env, bufsize=1,
    )
    proc.stdin.write(" ".join(str(i) for i in ids))
    proc.stdin.close()

    # Stream tokens: decode incrementally and print the delta. BPE tokens can
    # span UTF-8 boundaries, so we decode the cumulative buffer each step and
    # emit only the new suffix.
    out_ids = []
    last_text = ""
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        tid = int(line)
        if tid in EOT_IDS_LLAMA3:
            break
        out_ids.append(tid)
        cur = tok.decode(out_ids)
        sys.stdout.write(cur[len(last_text):])
        sys.stdout.flush()
        last_text = cur
    proc.wait()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
