import argparse
import torch

try:
    import triton
    import triton.language as tl
except Exception as e:
    raise SystemError("Triton is not installed or failed to import. Try: pip install -U triton") from e

try:
    from torch.cuda import nvtx  # NVTX markers shown in nsys timeline (if you profile on real CUDA box)
except Exception:
    nvtx = None

# ---------------------------
# Triton kernel: c = a + b
# ---------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

# ---------------------------
# Python wrapper
# ---------------------------
def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.device.type == "cuda" and y.device.type == "cuda", "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.dtype == y.dtype, "x and y must have the same dtype"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out

# ---------------------------
# Benchmark utility
# ---------------------------
def benchmark(n: int, iters: int, dtype=torch.float32, block_size: int = 1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemError("CUDA GPU required for Triton. Switch Colab to a GPU runtime.")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Elements: {n:,}, Iters: {iters}, Dtype: {dtype}, BLOCK_SIZE: {block_size}")

    a = torch.rand(n, dtype=dtype, device=device)
    b = torch.rand(n, dtype=dtype, device=device)

    # Warm-up (PyTorch + Triton)
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
    # 用 parse_known_args 避免 Colab 傳入的 -f 參數
    args, _unknown = parser.parse_known_args()

    dmap = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    benchmark(args.num, args.iters, dtype=dmap[args.dtype], block_size=args.block_size)
