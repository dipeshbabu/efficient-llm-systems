# TurboQuant on Windows — AMD RX 9070 XT (RDNA 4) Setup Guide

> First-time setup log for building llama.cpp with TurboQuant KV cache compression
> on Windows 11 with an AMD RX 9070 XT (gfx1201, RDNA 4).
>
> **Status**: WIP. RDNA 4 HIP support is bleeding-edge. Expect rough edges.

---

## Prerequisites

| Tool | Version | Install Method | Notes |
|------|---------|---------------|-------|
| Git | 2.x | Pre-installed | `C:\Program Files\Git` |
| Python | 3.10 | Windows Store | For the TurboQuant reference component |
| **VS 2022 Build Tools** | v143 | `winget install Microsoft.VisualStudio.2022.BuildTools` | **REQUIRED** — VS 2019 won't work (see Gotcha #1) |
| CMake | 4.3.1 | `winget install Kitware.CMake` | Installs to `C:\Program Files\CMake\bin` |
| Ninja | latest | `pip install ninja` or `winget install Ninja-build.Ninja` | pip version is more reliable for PATH |
| HIP SDK for Windows | 7.1 | Manual download from AMD | ~1.6GB, installs to `C:\Program Files\AMD\ROCm\7.1` |

---

## Step-by-Step Build

### 1. Install prerequisites

```powershell
# CMake
winget install Kitware.CMake

# Ninja (via pip — more reliable than winget for PATH)
pip install ninja

# VS 2022 Build Tools (MUST be 2022, not 2019)
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

### 2. Install HIP SDK for Windows

Download from AMD: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
(~1.6GB). Install to default location (`C:\Program Files\AMD\ROCm\7.1`).

Verify:
```cmd
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
hipcc --version
hipinfo
```

### 3. Clone repos

```bash
cd C:\models

# Python research prototype
git clone https://github.com/dipeshbabu/efficient-llm-systems.git
cd efficient-llm-systems && python -m venv .venv
# On Windows: source .venv/Scripts/activate  (NOT bin/activate)
source .venv/Scripts/activate
pip install -e ".[dev]"
python -m pytest tests/ -v  # 542+ pass, ~9 platform-specific failures OK

# llama.cpp TurboQuant fork (with HIP/ROCm support)
git clone --branch feature/turboquant-kv-cache \
  https://github.com/dipeshbabu/llama-cpp-turboquant.git llama-cpp-tq
```

### 4. Apply Windows HIP patches

See the gotchas below. Key changes needed:
- `ggml/src/ggml-hip/CMakeLists.txt` — add `-xhip -include __clang_hip_runtime_wrapper.h`
- `ggml/src/ggml-cuda/vendors/hip.h` — add `<algorithm>` + `using std::min/max`
- `ggml/src/ggml-turbo-quant.c` — add `M_PI` define
- `ggml/src/ggml-cuda/turbo-innerq.cu` — add `dllexport` on Windows
- `src/llama-kv-cache.cpp` — add `dllimport` on Windows
- `ggml/src/ggml-cuda/fattn-tile.cu` — guard D>=576 for HIP

### 5. Build

From **x64 Native Tools Command Prompt for VS 2022**:

```cmd
cd C:\models\llama-cpp-tq

set PATH=C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\CMake\bin;%PATH%
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1

cmake -S . -B build -G Ninja ^
    -DGPU_TARGETS=gfx1201 ^
    -DGGML_HIP=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### 6. Test inference

```cmd
:: Qwen2.5-7B Q4_K_M — use asymmetric K/V (recommended for Q4_K_M)
build\bin\llama-cli.exe ^
    -m C:\models\qwen2.5-7b-instruct-q4_k_m.gguf ^
    -ngl 99 -c 2048 -fa on ^
    --cache-type-k q8_0 --cache-type-v turbo4 ^
    -n 100 -p "Hello, I am a language model running on"
```

### First Run Results

```
Device 0: AMD Radeon RX 9070 XT, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 16304 MiB

Qwen2.5-7B-Instruct Q4_K_M, -ctk q8_0 -ctv turbo4, -c 512
Prompt:     412.0 t/s
Generation:  52.3 t/s
```

gfx1201 detected natively — no `HSA_OVERRIDE_GFX_VERSION` needed!

---

## Gotchas

### Gotcha #1: VS 2019 does NOT work — you need VS 2022

The `common/` library uses jinja templates that require a `<functional>` header with
C++17/20 `reinterpret_cast` semantics that VS 2019's v14.28 MSVC headers don't support.
HIP SDK's clang is stricter than MSVC cl.exe and rejects the VS 2019 code:

```
functional:1259:16: error: reinterpret_cast from 'const void *' to '...' casts away qualifiers
```

**Fix**: Install VS 2022 Build Tools with the `winget` command above.

### Gotcha #2: HIP SDK 7.1 device math broken on Windows (fabsf/fmaxf/expf/etc.)

MSVC's `<corecrt_math.h>` declares `fabsf`, `fmaxf`, `expf`, `logf`, `powf` as
`__inline` host-only functions. HIP's clang respects these attributes and refuses to
call them from `__device__` code.

On Linux this works because glibc's math headers don't have host/device annotations.
On Windows, the HIP device math wrappers in `__clang_hip_math.h` are only included
when compiling with `-xhip` flag — but llama.cpp's Windows HIP build compiles `.cu`
files as CXX (C++), not HIP language.

**Fix**: Add `-xhip -include __clang_hip_runtime_wrapper.h` to the compile flags for
all .cu files on Windows:

```cmake
# In ggml/src/ggml-hip/CMakeLists.txt
if (WIN32)
    set(HIP_WIN_FLAGS "-xhip -include __clang_hip_runtime_wrapper.h")
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "${HIP_WIN_FLAGS}")
endif()
```

### Gotcha #3: HIP SDK 7.1 complex builtins need min/max

`__clang_cuda_complex_builtins.h` (included via hip_runtime.h) uses bare `min`/`max`
which aren't defined in device scope on Windows.

**Fix**: Include `<algorithm>` and `using std::min; using std::max;` in `vendors/hip.h`
before `<hip/hip_runtime.h>`.

### Gotcha #4: M_PI not defined with clang + MSVC headers

The turbo-quant C code uses `M_PI` which MSVC only defines if `_USE_MATH_DEFINES`
is set before `<cmath>`.

**Fix**: Add `#ifndef M_PI / #define M_PI ...` after includes in `ggml-turbo-quant.c`.

### Gotcha #5: Cross-DLL symbol visibility (dllexport/dllimport)

llama.cpp builds as multiple DLLs on Windows (ggml-base.dll, ggml-cpu.dll,
ggml-hip.dll, llama.dll). Global variables and functions shared across DLLs need
`__declspec(dllexport)` on the definition side and `__declspec(dllimport)` on the
consumer side, with `extern "C"` linkage to avoid C++ name mangling.

The TurboQuant fork has three cross-DLL symbols that need this treatment:
- `turbo3_cpu_wht_group_size` (ggml-base → ggml-cpu)
- `g_innerq_scale_inv_host[]`, `turbo_innerq_needs_tensor_update()`,
  `turbo_innerq_mark_tensor_updated()` (ggml-hip → llama)

**Fix**: Create proper API functions with `GGML_API` decoration, or add
platform-specific `dllexport`/`dllimport` guards.

### Gotcha #6: Ninja PATH not available in Git Bash

`winget install Ninja-build.Ninja` adds to Windows PATH, but Git Bash doesn't
pick it up until you open a new terminal.

**Fix**: `pip install ninja` provides a Python-managed ninja that's always on PATH.

### Gotcha #7: D>=576 tile FA kernels exceed HIP local memory

The HIP CMakeLists excludes `fattn-tile` instances for D=576 and D=640 (exceed 65536
byte local memory limit). But the dispatch code in `fattn-tile.cu` still references
them, causing linker errors.

**Fix**: Guard the dispatch cases with `#ifdef GGML_USE_HIP` → `GGML_ABORT(...)`.

### Gotcha #8: HIP_PATH trailing space

When setting `HIP_PATH` in cmd.exe, beware of trailing spaces:
```cmd
set HIP_PATH=C:\Program Files\AMD\ROCm\7.1
```
The hipconfig script concatenates paths using `HIP_PATH` and a trailing space causes
"version file not found" warnings. Harmless but confusing.

### Gotcha #9: RDNA 4 (gfx1201) target — it works!

HIP SDK 7.1 (clang 21) includes gfx1201 support. No need for `HSA_OVERRIDE_GFX_VERSION`.
The GPU is detected correctly by `hipinfo`:
```
Name: AMD Radeon RX 9070 XT
warpSize: 32
totalGlobalMem: 15.92 GB
```

---

## Architecture Notes

The TurboQuant HIP implementation reuses CUDA `.cu` files compiled through the HIP
backend (standard AMD approach). Key files:

- `ggml/src/ggml-cuda/turbo-quant.cuh` — quantize/dequantize kernels
- `ggml/src/ggml-cuda/turbo-wht.cu` — Walsh-Hadamard Transform
- `ggml/src/ggml-cuda/turbo-innerq.cu` — per-channel equalization
- `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-turbo*.cu` — FlashAttention
- `ggml/src/ggml-hip/CMakeLists.txt` — HIP build configuration
