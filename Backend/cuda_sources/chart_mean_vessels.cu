#include "chart_mean_vessels.cuh"

#include "cuda_runtime.h"
#include "cuda_memory.cuh"
#include "hardware_limits.hh"

__global__ void get_sum_with_mask_kernel(const float* input, const float* mask, size_t size, float* sum_res)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        atomicAdd(sum_res + ((int)mask[idx]) * 2, 1);
        atomicAdd(sum_res + ((int)mask[idx]) * 2 + 1, input[idx]);
    }
}

holovibes::ChartMeanVesselsPoint
get_sum_with_mask(const float* input, const float* mask, size_t size, float* sum_res, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    float sum_res_host[8];

    cudaXMemset(sum_res, 0.0f, sizeof(float) * 8);
    get_sum_with_mask_kernel<<<blocks, threads, 0, stream>>>(input, mask, size, sum_res);
    cudaStreamSynchronize(stream);

    cudaXMemcpy(&sum_res_host, sum_res, sizeof(float) * 8, cudaMemcpyDeviceToHost);

    return {
        sum_res_host[3] / sum_res_host[2],
        sum_res_host[5] / sum_res_host[4],
        sum_res_host[7] / sum_res_host[6],
    };
}
