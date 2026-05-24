#!/usr/bin/env bash
# Compare smlx vs mlx-lm greedy throughput on the same prompt and decode length.
# Usage: bench.sh <model_dir> <max_new>
set -euo pipefail

MODEL="${1:-models/Llama-3.2-1B-Instruct-bf16}"
MAX_NEW="${2:-128}"

# A modest mixed-content prompt (~25 tokens after chat template).
PROMPT="Explain in one paragraph how attention works inside a transformer."

# Tokenize the chat-templated prompt with the tokenizer, dump ids.
IDS=$(python3 -c "
from tokenizers import Tokenizer
t = Tokenizer.from_file('$MODEL/tokenizer.json')
text = (
  '<|begin_of_text|>'
  '<|start_header_id|>user<|end_header_id|>\n\n'
  '$PROMPT'
  '<|eot_id|>'
  '<|start_header_id|>assistant<|end_header_id|>\n\n'
)
print(' '.join(str(i) for i in t.encode(text, add_special_tokens=False).ids))
")
N_PROMPT=$(echo "$IDS" | wc -w | tr -d ' ')
echo "prompt tokens: $N_PROMPT  | max_new: $MAX_NEW"
echo

echo "=== smlx (C / mlx-c, greedy) ==="
echo "$IDS" | ./build/smlx "$MODEL/smlx.config.txt" "$MODEL/model.safetensors" "$MAX_NEW" \
  > /tmp/smlx_out.txt 2> /tmp/smlx_err.txt
grep '\[smlx\]' /tmp/smlx_err.txt
echo

echo "=== mlx-lm (Python, greedy) ==="
python3 scripts/bench_mlx_lm.py "$MODEL" "$MAX_NEW" "$IDS"
