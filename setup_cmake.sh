#!/usr/bin/env bash
# Configure and build the C++ instantvnr libraries and executables directly
# via CMake, without going through Python packaging.
#
# Usage:
#   ./setup_cmake.sh               # auto-detect everything
#   SM=86 ./setup_cmake.sh         # override GPU arch
#   BUILD_DIR=build ./setup_cmake.sh  # custom build directory
#   ./setup_cmake.sh --configure   # configure only (skip build)
#   ./setup_cmake.sh --build       # build only (skip configure)
#
# Requires:
#   - CUDA toolkit (nvcc in PATH or /usr/local/cuda)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"

DO_CONFIGURE=true
DO_BUILD=true
for arg in "$@"; do
  case "$arg" in
    --configure) DO_BUILD=false ;;
    --build)     DO_CONFIGURE=false ;;
  esac
done

# ── detect GPU SM ─────────────────────────────────────────────────────────────
if [[ -n "${SM:-}" ]]; then
    echo "[info] Using SM=$SM from environment"
elif command -v nvidia-smi &>/dev/null; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "[info] Detected GPU sm_$SM"
else
    echo "[warn] nvidia-smi not found — defaulting to native arch detection"
    SM="native"
fi

# ── detect CUDA toolkit ───────────────────────────────────────────────────────
if   command -v nvcc &>/dev/null;       then CUDA_HOME="$(realpath "$(dirname "$(command -v nvcc)")/..")"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then CUDA_HOME="/usr/local/cuda"
else
    echo "[error] nvcc not found — CUDA toolkit is required" >&2
    exit 1
fi
echo "[info] CUDA_HOME: $CUDA_HOME"

export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── configure ─────────────────────────────────────────────────────────────────
if [[ "$DO_CONFIGURE" == true ]]; then
    echo "[info] Configuring in $BUILD_DIR (SM=$SM)"
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$SM"
fi

# ── build ─────────────────────────────────────────────────────────────────────
if [[ "$DO_BUILD" == true ]]; then
    JOBS="${JOBS:-$(nproc)}"
    echo "[info] Building with $JOBS parallel jobs"
    cmake --build "$BUILD_DIR" --config Release -- -j"$JOBS"
    echo "[info] Build complete. Outputs in $BUILD_DIR/instantvnr/"
fi
