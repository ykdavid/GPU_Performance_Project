# triton_addition.py
# Requirements:
#   pip install torch triton
# Hardware:
#   NVIDIA GPU + recent CUDA drivers (Triton JITs to PTX via CUDA)

import argparse
import torch
import triton
import triton.language as tl

try:
    from torch.cuda import nvtx
except Exception:
    nvtx = None


# ---------------------------
# Triton kernel: c = a + b
# ---------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                      # program id along the 1D grid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # element indices this program will handle
    mask = offsets < n_elements                 # guard against out-of-bounds
    x = tl.load(x_ptr + offsets, mask=mask)     # load a chunk of x
    y = tl.load(y_ptr + offsets, mask=mask)     # load a chunk of y
    tl.store(out_ptr + offsets, x + y, mask=mask)


# ---------------------------
# Python wrapper
# ---------------------------
def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Adds two CUDA tensors using Triton.
    Shapes and dtypes must match. Returns a tensor with the same dtype/shape/device.
    """
    assert x.device.type == "cuda" and y.device.type == "cuda", "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.dtype == y.dtype, "x and y must have the same dtype"

    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    # grid is the number of Triton programs to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


# ---------------------------
# Benchmark utility
# ---------------------------
def benchmark(n: int, iters: int, dtype=torch.float32, block_size: int = 1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemError("CUDA GPU required for Triton. Run this on a machine with an NVIDIA GPU.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Elements: {n:,}, Iters: {iters}, Dtype: {dtype}, BLOCK_SIZE: {block_size}")

    a = torch.rand(n, dtype=dtype, device=device)
    b = torch.rand(n, dtype=dtype, device=device)

    # Warm-up both PyTorch and Triton paths
    if nvtx: nvtx.range_push("warmup")
    _ = a + b
    _ = triton_add(a, b, block_size=block_size)
    torch.cuda.synchronize()
    if nvtx: nvtx.range_pop()

    # --- Time PyTorch (baseline) ---
    if nvtx: nvtx.range_push("pytorch_add")
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        _ = a + b
    end.record()
    torch.cuda.synchronize()
    pyt_time = start.elapsed_time(end) / 1000.0 / iters
    if nvtx: nvtx.range_pop()

    # --- Time Triton ---
    if nvtx: nvtx.range_push("triton_add")
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        _ = triton_add(a, b, block_size=block_size)
    end.record()
    torch.cuda.synchronize()
    tri_time = start.elapsed_time(end) / 1000.0 / iters
    if nvtx: nvtx.range_pop()

    # Bandwidth estimate: read a + read b + write out
    bytes_per_elem = a.element_size()
    bw_pt = 3 * n * bytes_per_elem / pyt_time / 1e9
    bw_tr = 3 * n * bytes_per_elem / tri_time / 1e9

    print(f"[PyTorch]  time/iter = {pyt_time:.6f} s,  bandwidth ≈ {bw_pt:.2f} GB/s")
    print(f"[Triton ]  time/iter = {tri_time:.6f} s,  bandwidth ≈ {bw_tr:.2f} GB/s")

    # Correctness check
    out_pt = a + b
    out_tr = triton_add(a, b, block_size=block_size)
    max_abs_err = (out_pt - out_tr).abs().max().item()
    print(f"Max abs error vs PyTorch: {max_abs_err:e}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10_000_000, help="Number of elements")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--block-size", type=int, default=1024, help="Triton BLOCK_SIZE (threads per program)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()

    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    benchmark(args.num, args.iters, dtype=dmap[args.dtype], block_size=args.block_size)
