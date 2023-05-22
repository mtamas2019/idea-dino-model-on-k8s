#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;

    // GPU Memory Test
    const int numElements = 1000000;
    const size_t size = numElements * sizeof(float);

    float* d_array;
    cudaMalloc(&d_array, size);

    if (d_array == nullptr) {
        std::cout << "Failed to allocate GPU memory" << std::endl;
        return 1;
    }

    cudaFree(d_array);

    return 0;
}
