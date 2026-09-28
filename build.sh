#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_PREFIX="$HOME/miniforge3/envs/vllm-dev"
export CONDA_PREFIX
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl${CPATH:+:$CPATH}"
export MAX_JOBS=8

cd "$REPO_ROOT"
"$CONDA_PREFIX/bin/python" -m pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm-build.log