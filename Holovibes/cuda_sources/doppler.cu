#include <stdio.h>
#include <iostream>
#include <fstream>

#include "doppler.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "unique_ptr.hh"
#include "logger.hh"

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

void compute_moments(const cuComplex* input,
                     float* gpu_moment_zero_out,
                     float* gpu_moment_two_out,
                     const ushort pmin,
                     const ushort pmax,
                     const size_t frame_res,
                     const cudaStream_t stream)
{
    auto func_moment_zero = [] __device__(size_t z) -> size_t { return 1; };
    auto func_moment_two = [] __device__(size_t z) -> size_t { return z * z; };

    compute_sum_depth(input, gpu_moment_zero_out, frame_res, pmin, pmax, func_moment_zero, stream);
    compute_sum_depth(input, gpu_moment_two_out, frame_res, pmin, pmax, func_moment_two, stream);
}

void fill_output_frame(
    float* output, float* gpu_moment_zero, float* gpu_moment_two, const size_t frame_res, const cudaStream_t stream)
{
    // Divide the moment2 by moment0 and put the result in the output_frame
    auto policy = thrust::cuda::par.on(stream);
    thrust::divides<float> op;
    thrust::transform(policy,                     // Execute on stream
                      gpu_moment_two,             // M2 begin
                      gpu_moment_two + frame_res, // M2 end
                      gpu_moment_zero,            // M0 begin
                      output,                     // Output frame begin
                      op);                        // Operation: divides
}

/// @brief Converts the complex buffer cube to single frame by using the doppler method
/// @param output The output float frame buffer
/// @param input The input complex buffer cube
/// @param gpu_moment_zero_out The gpu buffer for storing moment zero
/// @param gpu_moment_two_out The gpu buffer for storing moment two
/// @param pmin Min z index (z1)
/// @param pmax Max z index (z2)
/// @param frame_res The input and output frame resolution (width * height)
/// @param stream The cuda stream used
void complex_to_doppler(float* output,
                        const cuComplex* input,
                        float* gpu_moment_zero_out,
                        float* gpu_moment_two_out,
                        const ushort pmin,
                        const ushort pmax,
                        const size_t frame_res,
                        const cudaStream_t stream)
{
    compute_moments(input, gpu_moment_zero_out, gpu_moment_two_out, pmin, pmax, frame_res, stream);

    fill_output_frame(output, gpu_moment_zero_out, gpu_moment_two_out, frame_res, stream);
}