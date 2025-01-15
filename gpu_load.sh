#!/bin/bash

# Create a simple CUDA program
cd /opt/parallelcluster/shared

cat << EOF > simple_gpu_load.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void simple_kernel(float *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000000; i++) {
            d_data[idx] = sinf(d_data[idx]);
        }
    }
}

int main() {
    int n = 1000000;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    // Set up timer
    time_t start_time = time(NULL);
    time_t current_time;
    const int MAX_RUNTIME = 300; // 5 minutes in seconds
    
    while(1) {
        simple_kernel<<<grid, block>>>(d_data, n);
        cudaDeviceSynchronize();
        
        // Check if 5 minutes have elapsed
        current_time = time(NULL);
        if (current_time - start_time >= MAX_RUNTIME) {
            printf("5 minutes elapsed. Stopping program.\n");
            break;
        }
    }
    
    // Clean up
    cudaFree(d_data);
    return 0;
}
EOF

# Compile the CUDA program
/usr/local/cuda-12.4/bin/nvcc -o simple_gpu_load simple_gpu_load.cu

# Run the program
chmod +x simple_gpu_load
./simple_gpu_load
