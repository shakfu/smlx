#!/usr/bin/env bash
# Populate thirdparty/ with the vendored dependencies (not committed to git).
# Pinned to known-good commits; bump these if you intend to track upstream.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p thirdparty
cd thirdparty

MLXC_COMMIT=fba4470b89073180056c9ea46c443051375f7399          # ml-explore/mlx-c (~0.6.0)
TOKENIZERS_COMMIT=c586c52f93f7b060753bd2388eb96a105cb7374d    # mlc-ai/tokenizers-cpp

# 1. mlx-c (mlx C API; pulls mlx core via CMake FetchContent at configure time)
if [ ! -d mlx-c/.git ]; then
  git clone https://github.com/ml-explore/mlx-c.git
  git -C mlx-c checkout "$MLXC_COMMIT"
fi

# 2. tokenizers-cpp (HF tokenizer; needs submodules msgpack + sentencepiece)
if [ ! -d tokenizers-cpp/.git ]; then
  git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git
  git -C tokenizers-cpp checkout "$TOKENIZERS_COMMIT"
  git -C tokenizers-cpp submodule update --init --recursive
fi

# 3. minja headers (chat-template rendering) -- header-only
if [ ! -f minja/minja.hpp ]; then
  rm -rf /tmp/minja-src
  git clone --depth 1 https://github.com/google/minja.git /tmp/minja-src
  mkdir -p minja
  cp /tmp/minja-src/include/minja/*.hpp minja/
  rm -rf /tmp/minja-src
fi

# 4. nlohmann/json single header (used by minja)
if [ ! -f nlohmann/json.hpp ]; then
  mkdir -p nlohmann
  curl -fsSL https://github.com/nlohmann/json/releases/latest/download/json.hpp \
    -o nlohmann/json.hpp
fi

echo "thirdparty/ ready: mlx-c, tokenizers-cpp, minja, nlohmann"
