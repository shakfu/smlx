#!/usr/bin/env bash
# Correctness test: smlx greedy decode must be bit-exact vs mlx_lm for every
# downloaded model. Works in token-id space (no tokenizer in the compared path);
# the prompt is tokenized per-model so the ids are valid for each tokenizer.
#
# Needs .venv with `transformers` + `mlx_lm` (only for the reference side).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BIN=./build/smlx
N=${N:-24}
PROMPT_TEXT=${PROMPT_TEXT:-"The capital of France is"}

if [ ! -x "$PY" ]; then
  echo "error: $PY not found (need a .venv with transformers + mlx_lm)"; exit 2
fi
if [ ! -x "$BIN" ]; then
  echo "error: $BIN not built (run: make build)"; exit 2
fi

MODELS=(
  Llama-3.2-1B-Instruct-bf16
  Llama-3.2-1B-Instruct-4bit
  Qwen2.5-0.5B-Instruct-4bit
  Qwen3-0.6B-4bit
  Qwen3-4B-MLX-4bit
)

fail=0
tested=0
for m in "${MODELS[@]}"; do
  d="models/$m"
  [ -d "$d" ] || { echo "SKIP $m (not downloaded)"; continue; }
  tested=$((tested+1))

  ids=$("$PY" -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$d')
print(' '.join(map(str, t.encode('$PROMPT_TEXT', add_special_tokens=False))))
" 2>/dev/null)
  if [ -z "$ids" ]; then echo "FAIL $m (tokenize failed)"; fail=1; continue; fi

  smlx=$(echo "$ids" | "$BIN" ids "$d/smlx.config.txt" "$d/model.safetensors" "$N" 2>/dev/null \
         | tr '\n' ' ' | xargs)
  ref=$("$PY" scripts/ref_qwen.py "$d" "$N" $ids 2>/dev/null | xargs)

  if [ -n "$smlx" ] && [ "$smlx" = "$ref" ]; then
    echo "PASS $m"
  else
    echo "FAIL $m"
    echo "  ids : $ids"
    echo "  smlx: $smlx"
    echo "  ref : $ref"
    fail=1
  fi
done

echo "---"
if [ "$tested" -eq 0 ]; then
  echo "no models found under models/ — nothing tested"; exit 2
fi
if [ "$fail" -eq 0 ]; then echo "all $tested model(s) bit-exact vs mlx_lm"; else echo "FAILURES above"; fi
exit $fail
