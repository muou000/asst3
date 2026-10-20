#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
// 辅助函数：将整数向上舍入为下一个 2 的幂次方
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
//
// exclusive_scan --
// 在全局内存数组 `input` 上实现排他性扫描 (exclusive scan)，并将结果放入全局内存 `result` 中。
//
// N 是输入和输出数组的逻辑大小，但学生可以假定 start 和 result 数组都是按照 cudaScan() 注释中所述
// 的下一个 2 的幂次方大小分配的。这很有用，因为你的并行 scan 可能会写入超过 N 的内存位置，
// 但当然不会超过 N 向上舍入后的下一个 2 的幂。
//
// 此外，根据 cudaScan() 中的注释，你可以实现“原地 (in-place)”扫描，因为测试框架会复制 input 并放入 result 中。
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    //
    // CS149 待办事项：
    // 在此处实现你的排他性扫描 (exclusive scan)。
    // 请记住：虽然此函数的参数是在设备上分配的数组，但此函数本身是在 CPU 线程上运行的。
    // 你的实现需要多次调用 CUDA 核函数（你需要自己编写这些核函数）来完成扫描。


}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
//
// cudaScan --
// 此函数是围绕学生实现的 scan 的计时包装器 —— 它将输入复制到 GPU，
// 并对上面 exclusive_scan() 函数的调用进行计时。学生不应修改此函数。
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.
    //
    // 此代码将提供给 exclusive_scan 的数组长度向上取整到 2 的幂，
    // 但超出原始输入末尾的元素保持未初始化状态，且不会检查其正确性。
    // 为简便起见，学生实现的 exclusive_scan 可以假定数组的分配长度为 2 的幂。
    // 这会在非 2 的幂输入上导致额外的计算工作，但换取仅支持 2 的幂解法的简洁性是值得的。

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    // 为方便起见，设备上的输入和输出向量都初始化为输入数组的值。
    // 这意味着如果需要，学生可以在 result 向量上自由实现原地扫描。
    // 如果你这样做，在从 find_repeats 调用 exclusive_scan 时需要记住这一点。
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion / 等待 GPU 完成
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
//
// cudaScanThrust --
// 对 Thrust 库 exclusive scan 函数的包装。
// 与上面的 cudaScan() 类似，此函数将输入复制到 GPU，且仅对 scan 本身的执行进行计时。
// 不要求学生实现的性能达到与 Thrust 版本竞争的水平，但尝试一下很有趣。
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
//
// find_repeats --
// 给定整数数组 `device_input`，返回满足 `device_input[i] == device_input[i+1]` 的所有下标 `i` 的数组。
// 返回找到的相邻重复对的总数。
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    //
    // CS149 待办事项：
    // 实现此函数。你可能需要使用一次或多次 exclusive_scan() 调用，以及额外的 CUDA 核函数启动。
    //
    // 注意：与 scan 代码中一样，调用方代码确保分配的数组大小为 2 的幂，
    // 因此你可以对它们使用你的 exclusive_scan 函数。
    // 但是，在给定实际数组长度的情况下，你的实现必须确保 find_repeats 的结果是正确的。

    return 0; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
// 对 find_repeats 的计时包装器。你不应修改此函数。
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array / 设置输出计数与结果数组
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
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
