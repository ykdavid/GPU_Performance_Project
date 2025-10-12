// cuda_addition.cu
// Build: nvcc -O3 -std=c++17 -arch=sm_70 cuda_addition.cu -o cuda_add
// Usage: ./cuda_add [N=100000000] [BLOCK=256] [dtype=float|int]
// Example: ./cuda_add 10000000 256 float

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

#define CHECK_CUDA(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        std::cerr << "CUDA error " << cudaGetErrorString(_e)       \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";\
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

template <typename T>
__global__ void add_kernel(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ c,
                           size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

template <typename T>
void run_case(size_t N, int BLOCK, int iters) {
    std::cout << "N=" << N << ", BLOCK=" << BLOCK << ", dtype=" 
              << (std::is_same<T,float>::value ? "float" : "int") << "\n";

    // Host input/output
    std::vector<T> h_a(N), h_b(N), h_c(N);
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<T>(i % 1024);
        h_b[i] = static_cast<T>(1);
    }

    // Device memory
    T *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = N * sizeof(T);
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // H2D
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 block(BLOCK);
    dim3 grid((N + block.x - 1) / block.x);

    // Warm-up
    add_kernel<T><<<grid, block>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        add_kernel<T><<<grid, block>>>(d_a, d_b, d_c, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float sec_per_iter = (ms / 1000.0f) / iters;

    // Bandwidth: read a + read b + write c
    double gbps = (3.0 * static_cast<double>(bytes) / sec_per_iter) / 1e9;

    std::cout << "Time per iteration: " << sec_per_iter << " s\n";
    std::cout << "Estimated bandwidth: " << gbps << " GB/s\n";

    // D2H + verify
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < std::min<size_t>(N, 1000); ++i) {
        T ref = h_a[i] + h_b[i];
        if (h_c[i] != ref) { ok = false; break; }
    }
    std::cout << "Correctness check: " << (ok ? "PASS" : "FAIL") << "\n";

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 100000000ULL;
    int BLOCK   = (argc > 2) ? std::atoi(argv[2]) : 256;
    std::string dtype = (argc > 3) ? argv[3] : "float";
    int iters = 100;  // number of timed launches

    int dev = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << "\n";

    if (dtype == "int")
        run_case<int>(N, BLOCK, iters);
    else
        run_case<float>(N, BLOCK, iters);

    return 0;
}
