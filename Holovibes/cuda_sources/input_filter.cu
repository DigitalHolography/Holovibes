#include "input_filter.cuh"

#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

void apply_filter(float* gpu_filter2d_mask,
                  float* gpu_input_filter_mask,
                  const float* input_filter,
                  size_t width,
                  size_t height,
                  const cudaStream_t stream)
{
    size_t frame_res = width * height;

    // copy the cpu input_filter into gpy input_filter buffer
    cudaXMemcpyAsync(gpu_input_filter_mask, input_filter, frame_res * sizeof(float), cudaMemcpyHostToDevice, stream);

    // gpu_filter2d_mask is already shifted, we only need to shift the input_filter
    shift_corners(gpu_input_filter_mask, 1, width, height, stream);

    // Element wise multiplies the two masks
    auto policy = thrust::cuda::par.on(stream);
    thrust::multiplies<float> op;
    thrust::transform(policy,                        // Execute on stream
                      gpu_filter2d_mask,             // Input1 begin
                      gpu_filter2d_mask + frame_res, // Input1 end
                      gpu_input_filter_mask,         // Input2 begin
                      gpu_filter2d_mask,             // Output begin
                      op);                           // Operation: multiplies
}