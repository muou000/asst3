#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"


// return GB/sec
// 返回 GB/秒
float GBPerSec(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
// 这是在 GPU 上运行的 CUDA "核函数 (kernel)"。
// 你可以通过它被标记为 __global__ 函数得知这一点。
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    // 根据线程在当前线程块中的位置以及所在的线程块，计算全局线程索引
    // （在当前示例中只需要 1D 计算，因此代码仅查看 blockDim 和 threadIdx 的 .x 项）。
    int index = blockIdx.x * blockDim.x + threadIdx.x;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    // 此检查是必要的，以使代码能够处理不是线程块大小 (blockDim.x) 整数倍的 N 值。
    if (index < N)
       result[index] = alpha * x[index] + y[index];
}


// saxpyCuda --
//
// This function is regular C code running on the CPU.  It allocates
// memory on the GPU using CUDA API functions, uses CUDA API functions
// to transfer data from the CPU's memory address space to GPU memory
// address space, and launches the CUDA kernel function on the GPU.
//
// saxpyCuda --
// 此函数是在 CPU 上运行的常规 C/C++ 代码。它使用 CUDA API 函数在 GPU 上分配显存，
// 使用 CUDA API 函数将数据从 CPU 的内存地址空间传输到 GPU 显存地址空间，
// 并在 GPU 上启动 CUDA 核函数。
void saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    // must read both input arrays (xarray and yarray) and write to
    // output array (resultarray)
    // 必须读取两个输入数组（xarray 和 yarray）并写入输出数组（resultarray）
    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block.  In this
    // application we've hardcoded thread blocks to contain 512 CUDA
    // threads.
    // 计算块数和每块的线程数。在此应用程序中，我们硬编码每个线程块包含 512 个 CUDA 线程。
    const int threadsPerBlock = 512;

    // Notice the round up here.  The code needs to compute the number
    // of threads blocks needed such that there is one thread per
    // element of the arrays.  This code is written to work for values
    // of N that are not multiples of threadPerBlock.
    // 注意这里的向上取整。代码需要计算所需的线程块数量，以便数组中的每个元素对应一个线程。
    // 该代码适用于不是 threadPerBlock 整数倍的 N 值。
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // These are pointers that will be pointers to memory allocated
    // *one the GPU*.  You should allocate these pointers via
    // cudaMalloc.  You can access the resulting buffers from CUDA
    // device kernel code (see the kernel function saxpy_kernel()
    // above) but you cannot access the contents these buffers from
    // this thread. CPU threads cannot issue loads and stores from GPU
    // memory!
    // 这些指针将指向在 *GPU* 上分配的显存。
    // 你应该通过 cudaMalloc 为这些指针分配空间。
    // 你可以从 CUDA 设备端核函数代码（见上面的核函数 saxpy_kernel()）中访问生成的缓冲区，
    // 但你无法从当前 CPU 线程访问这些缓冲区的内容。CPU 线程无法直接对 GPU 显存执行读取和写入！
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_result = nullptr;

    //
    // CS149 TODO: allocate device memory buffers on the GPU using cudaMalloc.
    // CS149 待办事项：使用 cudaMalloc 在 GPU 上分配设备显存缓冲区。
    //
    // We highly recommend taking a look at NVIDIA's
    // tutorial, which clearly walks you through the few lines of code
    // you need to write for this part of the assignment:
    // 我们强烈建议查看 NVIDIA 的教程，它清晰地引导你完成本部分作业所需编写的几行代码：
    //
    // https://devblogs.nvidia.com/easy-introduction-cuda-c-and-c/
    //
    cudaMalloc(&device_x, sizeof(float) * N);
    cudaMalloc(&device_y, sizeof(float) * N);
    cudaMalloc(&device_result, sizeof(float) * N);

    // start timing after allocation of device memory
    // 设备显存分配完成后开始计时
    double startTime = CycleTimer::currentSeconds();

    //
    // CS149 TODO: copy input arrays to the GPU using cudaMemcpy
    // CS149 待办事项：使用 cudaMemcpy 将输入数组复制到 GPU
    //
    cudaMemcpy(device_x, xarray, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, sizeof(float) * N, cudaMemcpyHostToDevice);

    // run CUDA kernel. (notice the <<< >>> brackets indicating a CUDA
    // kernel launch) Execution on the GPU occurs here.
    // 运行 CUDA 核函数。（注意指示 CUDA 核函数启动的 <<< >>> 括号语法）
    // GPU 上的执行在此发生。
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    //
    // CS149 TODO: copy result from GPU back to CPU using cudaMemcpy
    // CS149 待办事项：使用 cudaMemcpy 将结果从 GPU 复制回 CPU
    //
    cudaMemcpy(resultarray, device_result, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    // 结果复制回主机内存后结束计时
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n",
		errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Effective BW by CUDA saxpy: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, GBPerSec(totalBytes, overallDuration));

    //
    // CS149 TODO: free memory buffers on the GPU using cudaFree
    // CS149 待办事项：使用 cudaFree 释放在 GPU 上的显存缓冲区
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.
    // 打印关于机器中 GPU 的统计信息。如果学生想了解他们在哪款 GPU 上运行，这很有用。

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
