#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 NEW_BUILD_WORKSPACE" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace=$1
if [ -e "$workspace" ]; then
    echo "Build workspace must be new; existing files are preserved." >&2
    exit 2
fi
mkdir -p -- "$workspace"
workspace=$(CDPATH= cd -- "$workspace" && pwd)
revision=434ddbbc0e30522e897670681e503b797c12b7c1
cmake_bin=${CMAKE:-cmake}

git init "$workspace/source"
git -C "$workspace/source" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "$workspace/source" fetch --depth 1 origin "$revision"
git -C "$workspace/source" checkout --detach "$revision"
git -C "$workspace/source" apply "$script_dir/llamacpp-capture.patch"
"$cmake_bin" -S "$workspace/source" -B "$workspace/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
    -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
"$cmake_bin" --build "$workspace/build" --target llama-completion -j 2
echo "Capture provider: $workspace/build/bin/llama-completion"
