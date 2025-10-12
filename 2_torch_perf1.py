import torch
import time

def add_tensors(a, b):
    # Element-wise addition of two tensors
    return a + b

# Automatically detect whether CUDA (GPU) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Number of elements in each tensor
num = 10_000_000

# Create random tensors on the selected device (CPU or GPU)
a = torch.rand(num, device=device)
b = torch.rand(num, device=device)

# Warm-up operation (important for CUDA initialization)
c = add_tensors(a, b)

# Start timing
t = time.time()

# Perform the tensor addition 100 times
for _ in range(100):
    c = add_tensors(a, b)

# End timing
end = time.time()

# Calculate the average time per iteration
every_iteration_time = (end - t) / 100
print(f"Time per iteration: {every_iteration_time:.8f} seconds")

# Compute effective memory bandwidth (GB/s)
# Each iteration reads a and b, and writes c -> total 3 * num * element_size() bytes
bandwidth = 3 * num * a.element_size() / every_iteration_time / 1e9
print(f"Bandwidth: {bandwidth:.3f} GB/s")

# Move result back to CPU for potential further processing or display
c = c.to('cpu')

# Print the first 10 elements of the resulting tensor
print(c[:10])
