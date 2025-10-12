import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def add_torch(a, b):
    # Element-wise tensor addition
    return a + b

# Problem size and iterations
num = 100_000_000
iters = 1000

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")

# Choose profiler activities based on availability
activities = [ProfilerActivity.CPU]
if use_cuda:
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, profile_memory=True) as prof:
    # Allocate inputs on CPU first (to measure transfer too if needed)
    a = torch.rand(num, device="cpu")
    b = torch.rand(num, device="cpu")

    # Move tensors to target device
    a = a.to(device)
    b = b.to(device)

    # Warm-up to ensure kernels are initialized
    result = add_torch(a, b)

    if use_cuda:
        # High-resolution GPU timing with CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with record_function("add_loop"):
            for _ in range(iters):
                result = add_torch(a, b)
        end.record()

        # Ensure all kernels complete before reading the timer
        torch.cuda.synchronize()

        # Per-iteration time (milliseconds -> seconds)
        per_iter_sec = (start.elapsed_time(end) / iters) / 1000.0
    else:
        # CPU timing fallback
        t0 = time.perf_counter()
        with record_function("add_loop"):
            for _ in range(iters):
                result = add_torch(a, b)
        per_iter_sec = (time.perf_counter() - t0) / iters

    print(f"Time per iteration: {per_iter_sec:.8f} seconds")

    # Effective memory bandwidth (GB/s):
    # Each iteration reads a and b and writes result -> 3 * num * element_size() bytes
    element_bytes = a.element_size()
    bandwidth_gbs = 3 * num * element_bytes / per_iter_sec / 1e9
    print(f"Bandwidth: {bandwidth_gbs:.3f} GB/s")

    # Bring a small slice back to CPU to verify correctness without huge prints
    print(result[:10].to("cpu"))

# Export Chrome/Perfetto trace; open at https://ui.perfetto.dev/ and load the file
prof.export_chrome_trace("trace.json")
print('Profiler trace saved to "trace.json" (open in Perfetto to inspect).')
