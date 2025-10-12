import torch

def add_tensors(a, b):
    # Element-wise tensor addition
    return a + b

# Make sure CUDA is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available on this system!")

device = torch.device('cuda')
print(f"Using device: {device}")

# Number of elements in each tensor
num = 10_000_000

# Create random tensors on CPU and transfer to GPU
a = torch.rand(num, device='cpu').to(device)
b = torch.rand(num, device='cpu').to(device)

# Warm-up operation (to initialize GPU kernels)
c = add_tensors(a, b)

# Create CUDA events for accurate GPU timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Record the start event
start.record()

# Perform 100 iterations of tensor addition on GPU
for _ in range(100):
    c = add_tensors(a, b)

# Record the end event
end.record()

# Wait for all kernels to complete before measuring time
torch.cuda.synchronize()

# Compute the average time per iteration in seconds
every_iteration_time = start.elapsed_time(end) / 1000 / 100
print(f"Time per iteration: {every_iteration_time:.8f} seconds")

# Compute effective memory bandwidth in GB/s
# Each iteration reads tensors a and b, and writes tensor c → 3 * num * element_size()
bandwidth = 3 * num * a.element_size() / every_iteration_time / 1e9
print(f"Bandwidth: {bandwidth:.3f} GB/s")

# Move result tensor back to CPU for display
c = c.to('cpu')

# Print the first 10 elements of the resulting tensor
print(c[:10])
