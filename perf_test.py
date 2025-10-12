
import argparse
import time
import torch

try:
    from torch.cuda import nvtx  # NVTX markers shown in nsys timeline
except Exception:
    nvtx = None  # CPU-only or older builds

def add_torch(a, b):
    # Element-wise addition
    return a + b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100_000_000, help="Number of elements")
    parser.add_argument("--iters", type=int, default=1000, help="Iterations for the loop")
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device} (CUDA available: {use_cuda})")
    print(f"num={args.num:,}, iters={args.iters}")

    # Allocate on CPU first then move to target device
    if nvtx: nvtx.range_push("alloc_cpu")
    a_cpu = torch.rand(args.num, device="cpu")
    b_cpu = torch.rand(args.num, device="cpu")
    if nvtx: nvtx.range_pop()

    if nvtx: nvtx.range_push("h2d")
    a = a_cpu.to(device, non_blocking=True)
    b = b_cpu.to(device, non_blocking=True)
    if use_cuda:
        torch.cuda.synchronize()
    if nvtx: nvtx.range_pop()

    # Optional warmup to stabilize kernels / cuBLAS init, etc.
    if not args.no_warmup:
        if nvtx: nvtx.range_push("warmup")
        _ = add_torch(a, b)
        if use_cuda:
            torch.cuda.synchronize()
        if nvtx: nvtx.range_pop()

    # Timed loop (CUDA events when possible for accuracy)
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        if nvtx: nvtx.range_push("compute_loop")
        start.record()
        for _ in range(args.iters):
            _ = add_torch(a, b)
        end.record()
        torch.cuda.synchronize()
        if nvtx: nvtx.range_pop()

        per_iter_sec = (start.elapsed_time(end) / args.iters) / 1000.0
    else:
        t0 = time.perf_counter()
        if nvtx: nvtx.range_push("compute_loop_cpu")
        for _ in range(args.iters):
            _ = add_torch(a, b)
        if nvtx: nvtx.range_pop()
        per_iter_sec = (time.perf_counter() - t0) / args.iters

    print(f"Time per iteration: {per_iter_sec:.8f} s")
    bandwidth_gbs = 3 * args.num * a.element_size() / per_iter_sec / 1e9
    print(f"Estimated bandwidth: {bandwidth_gbs:.3f} GB/s")

    if nvtx: nvtx.range_push("d2h")
    res_head = _.to("cpu")[:10]
    if use_cuda:
        torch.cuda.synchronize()
    if nvtx: nvtx.range_pop()

    print(res_head)

if __name__ == "__main__":
    main()
