#include "input_filter.cuh"

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void apply_filter(float* gpu_filter, cuComplex* gpu_input, size_t frame_res, const cudaStream_t stream)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    auto mult_func = []  __device__ (cuComplex a, float b) {return make_cuComplex(a.x * b, a.y * b)};

    thrust::transform(exec_policy, gpu_input, gpu_input + frame_res, gpu_filter, mult_func);
}