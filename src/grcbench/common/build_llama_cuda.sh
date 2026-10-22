#!/usr/bin/env bash
# Build the pinned llama.cpp b10333 with CUDA for the RTX 5090 (compute capability 12.0) inside WSL,
# with a user-space toolchain (micromamba, conda-forge): no sudo, nothing outside ~/grc-bench-cache.
# Run by `./gb fetch --cuda`. Upstream publishes CUDA builds of b10333 for Windows only, so the tag's
# source (commit 08659901c) is compiled here with CUDA 12.9 and GCC 13; the binary reports
# "version: 10333 (08659901c)" like the official build. Output: ~/grc-bench-cache/llama.cpp/b10333-cuda
# (llama-server, llama-bench, llama-cli and the libraries they need), used by `./gb tier-a smoke
# --backend cuda` and `./gb tier-e --llama-dir`. The last step runs llama-bench on the GPU as a check.
set -euo pipefail
CACHE="$HOME/grc-bench-cache"
TOOLS="$CACHE/tools"
ENVP="$CACHE/buildenv-cuda"
SRC="$CACHE/llama.cpp/src-b10333"
OUT="$CACHE/llama.cpp/b10333-cuda"
MODEL="$CACHE/models/Qwen3-4B-Q4_K_M.gguf"
export MAMBA_ROOT_PREFIX="$CACHE/mamba"
mkdir -p "$TOOLS"

echo "== 1. micromamba"
if [ ! -x "$TOOLS/bin/micromamba" ]; then
  curl -Ls -o "$TOOLS/micromamba.tar.bz2" https://micro.mamba.pm/api/micromamba/linux-64/latest
  python3 -c "import sys, tarfile; tarfile.open(sys.argv[1], 'r:bz2').extract('bin/micromamba', sys.argv[2], filter='data')" \
    "$TOOLS/micromamba.tar.bz2" "$TOOLS"
  chmod +x "$TOOLS/bin/micromamba"
  rm -f "$TOOLS/micromamba.tar.bz2"
fi
"$TOOLS/bin/micromamba" --version

echo "== 2. toolchain"
if [ ! -x "$ENVP/bin/nvcc" ]; then
  "$TOOLS/bin/micromamba" create -y -q -p "$ENVP" -c conda-forge \
    cmake ninja "gcc_linux-64=13" "gxx_linux-64=13" "cuda-version=12.9" cuda-nvcc cuda-cudart-dev libcublas-dev
fi
"$ENVP/bin/nvcc" --version | tail -2

echo "== 3. source b10333"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  mkdir -p "$SRC"
  curl -Ls https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10333.tar.gz | tar -xz -C "$SRC" --strip-components=1
fi
sha256sum "$SRC/CMakeLists.txt"

echo "== 4. build"
set +u                                   # conda's activation scripts read unset variables
eval "$("$TOOLS/bin/micromamba" shell hook -s bash)"
micromamba activate "$ENVP"
set -u
echo "CC=${CC:-} CXX=${CXX:-} CUDAHOSTCXX=${CUDAHOSTCXX:-}"
cmake -S "$SRC" -B "$SRC/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_NUMBER=10333 -DLLAMA_BUILD_COMMIT=08659901c \
  > "$SRC/cmake-configure.log" 2>&1 || { tail -40 "$SRC/cmake-configure.log"; exit 1; }
cmake --build "$SRC/build-cuda" --target llama-server llama-bench llama-cli -j 20 > "$SRC/cmake-build.log" 2>&1 \
  || { tail -60 "$SRC/cmake-build.log"; exit 1; }
set +u
micromamba deactivate
set -u

echo "== 5. install to $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
cp -a "$SRC/build-cuda/bin/llama-server" "$SRC/build-cuda/bin/llama-bench" "$SRC/build-cuda/bin/llama-cli" "$OUT/"
cp -a "$SRC"/build-cuda/bin/*.so* "$OUT/" 2>/dev/null || true
for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libgomp.so.1; do
  for f in "$ENVP"/lib/"$lib"*; do [ -e "$f" ] && cp -a "$f" "$OUT/"; done
done
ls "$OUT"
export LD_LIBRARY_PATH="$OUT:/usr/lib/wsl/lib"
ldd "$OUT/llama-server" | grep -E "not found|cuda|cublas" || true
"$OUT/llama-server" --version 2>&1 | tail -3

echo "== 6. quick check on the GPU"
"$OUT/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 64 -r 2 2>&1 | tail -8
echo "BUILD OK"
