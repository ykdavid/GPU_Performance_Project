# GPU_Performance_Project
# GPU Vector Addition — Multi-Method Demo

This repo shows **eight ways** to implement and profile elementwise vector addition on CPU/GPU:

* Pure PyTorch (CPU/GPU)
* PyTorch + simple timing
* PyTorch + CUDA event timing
* PyTorch + `torch.profiler` trace
* PyTorch + **Nsight Systems** (`nsys`)
* PyTorch + **Nsight Compute** (`ncu`)
* **Triton** kernel
* **CUDA C++** kernel

---

## 0) Prerequisites

### Hardware / OS

* NVIDIA GPU + recent driver (for CUDA paths)
* Linux (recommended) or WSL2; macOS works for CPU-only parts

### Python env

```bash
# Example using conda (recommended)
conda create -n gpu-add python=3.11 -y
conda activate gpu-add

# PyTorch (choose wheel matching your CUDA)
# Example: CUDA 12.1 wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Triton (for 7_triton_addition.py)
pip install triton
```

### Nsight tools (optional, for 5 & 6)

* **Nsight Systems** (`nsys`) and **Nsight Compute** (`ncu`)

  * On Ubuntu via CUDA Toolkit or separate installers from NVIDIA.
  * Verify:

    ```bash
    nsys --version
    ncu --version
    ```

---

## 1) File index

| File                       | Technique                   | What it demonstrates                              |
| -------------------------- | --------------------------- | ------------------------------------------------- |
| `1_torch_addition.py`      | PyTorch + device fallback   | Minimal tensor add with auto CPU/GPU selection    |
| `2_torch_perf1.py`         | PyTorch + wall-clock timing | Loop timing with `time.time()`                    |
| `3_torch_perf1.py`         | PyTorch + CUDA events       | Accurate GPU timing with `torch.cuda.Event`       |
| `4_torch_torch_profile.py` | PyTorch Profiler            | CPU+CUDA trace, export to Perfetto (`trace.json`) |
| `5_torch_nsys.py`          | PyTorch + Nsight Systems    | How to run under `nsys profile`                   |
| `6_torch_ncu.py`           | PyTorch + Nsight Compute    | How to run under `ncu` and collect metrics        |
| `7_triton_addition.py`     | Triton kernel               | Custom JIT kernel + baseline compare + bandwidth  |
| `8_cuda_addition.cu`       | CUDA C++ kernel             | Hand-written kernel, events, bandwidth, verify    |

> Note: Some filenames in your screenshots used “perf2”; here they’re normalized as shown above.

---

## 2) Quick start (PyTorch baseline)

```bash
python 1_torch_addition.py
```

Expected:

* Prints selected device
* Performs `a + b`
* Shows first few elements (or summary)

If you see:

```
AssertionError: Torch not compiled with CUDA enabled
```

your wheel is CPU-only. Either:

* Let the scripts fall back to CPU, or
* Install a CUDA-enabled wheel (see prerequisites).

---

## 3) Scripts & commands

### A) `1_torch_addition.py` — Minimal add with device fallback

* Auto-selects `cuda` if available else `cpu`
* Creates two random tensors, adds them, prints a small slice

Run:

```bash
python 1_torch_addition.py
```

---

### B) `2_torch_perf1.py` — Wall-clock timing

* Uses `time.time()` (CPU or GPU)
* Reports **average time/iteration** and **estimated bandwidth**

Run:

```bash
python 2_torch_perf1.py
```

---

### C) `3_torch_perf1.py` — CUDA event timing (accurate on GPU)

* Uses `torch.cuda.Event(enable_timing=True)`
* Synchronizes with `torch.cuda.synchronize()`
* Reports **time/iter** and **bandwidth**

Run (requires CUDA):

```bash
python 3_torch_perf1.py
```

---

### D) `4_torch_torch_profile.py` — PyTorch Profiler trace

* Uses `torch.profiler.profile(activities=[CPU, CUDA], profile_memory=True)`
* Wraps compute section with `record_function` for readable labels
* Exports Perfetto trace: `trace.json`

Run:

```bash
python 4_torch_torch_profile.py
```

Open trace:

* Go to [https://ui.perfetto.dev](https://ui.perfetto.dev)
* “Open trace file” → select `trace.json`

---

### E) `5_torch_nsys.py` — Nsight Systems (system-level timeline)

The script itself is a normal PyTorch add loop with **NVTX ranges** to make the timeline readable. Run it **through `nsys`**:

```bash
nsys profile -o torch_add \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  python 5_torch_nsys.py
```

* Output: `torch_add.qdrep`
* View: `nsys-ui` (GUI) → open `.qdrep`
* Quick summary:

  ```bash
  nsys stats torch_add.qdrep
  ```

You’ll see CUDA API calls, kernel launches, memcpy, CPU threads, and NVTX regions.

---

### F) `6_torch_ncu.py` — Nsight Compute (kernel-level metrics)

Run **through `ncu`** to collect per-kernel metrics like SM occupancy, memory throughput, warp efficiency:

```bash
# full set (slower but thorough)
ncu --set full --target-processes all \
  python 6_torch_ncu.py
```

* Output: `.ncu-rep` in current dir (or specify `-o name`)
* View: `ncu-ui` (GUI), or rely on terminal tables

Optional filters:

```bash
# Focus on PyTorch add kernels & skip warmup launches
ncu --set roofline --target-processes all \
  --kernel-name ".*add.*" --launch-skip 5 --launch-count 50 \
  python 6_torch_ncu.py
```

---

### G) `7_triton_addition.py` — Triton kernel + baseline compare

* Implements an elementwise `add_kernel` in Triton
* Wrapper `triton_add(...)`
* Benchmarks vs PyTorch with CUDA events
* Prints **time/iter**, **bandwidth**, and **max abs error**

Run:

```bash
# fp32, 10M elems, 100 iters, BLOCK_SIZE=1024
python 7_triton_addition.py --num 10000000 --iters 100 --block-size 1024 --dtype fp32
```

Profile with tools:

```bash
nsys profile -o triton_add --trace=cuda,nvtx,osrt \
  python 7_triton_addition.py --num 10000000 --iters 200

ncu --set full --target-processes all \
  python 7_triton_addition.py --num 10000000 --iters 50
```

---

### H) `8_cuda_addition.cu` — CUDA C++ implementation

* Hand-written `__global__` kernel
* Launch config: `grid((N + BLOCK - 1) / BLOCK), block(BLOCK)`
* Accurate timing with CUDA events
* Bandwidth + correctness check

Build:

```bash
# Change sm_XX to your GPU (e.g., sm_80 for A100, sm_86 for 30xx, sm_90 for H100)
nvcc -O3 -std=c++17 -arch=sm_80 8_cuda_addition.cu -o cuda_add
```

Run:

```bash
./cuda_add 10000000 256 float   # N=10M, block=256, dtype=float|int
```

Profile:

```bash
# Nsight Systems
nsys profile -o cuda_add --trace=cuda,osrt ./cuda_add 10000000 256 float

# Nsight Compute
ncu --set full ./cuda_add 10000000 256 float
```

---

## 4) Bandwidth model used

For an elementwise add `c = a + b`, each iteration moves:

* Read **A** + Read **B** + Write **C** → **3 × N × element_size** bytes

Estimated bandwidth:

```
GB/s = (3 * N * element_size) / (time_per_iteration) / 1e9
```

---

## 5) Troubleshooting

* **`Torch not compiled with CUDA enabled`**

  * Install a CUDA-enabled wheel (see PyTorch install step), or let scripts run on CPU.

* **`CUDA driver version is insufficient`**

  * Update NVIDIA driver to match your CUDA runtime.

* **`nsys`/`ncu` not found**

  * Install Nsight tools; verify with `nsys --version` / `ncu --version`.

* **Slow first iteration**

  * GPU warm-up is intentional; use multiple iterations and average.

---

## 6) Suggested experiments

* Sweep block sizes (`128/256/512/1024`) for Triton and CUDA C++.
* Compare dtypes (`fp16`, `bf16`, `fp32`) and note bandwidth differences.
* Use `ncu` roofline view to see if you’re **memory-bound** or **compute-bound**.
* Check `nsys` timeline for H2D/D2H overlap and kernel launch gaps.
