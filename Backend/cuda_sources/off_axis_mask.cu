#include "off_axis_mask.cuh"

#include "cuda_memory.cuh"
#include "tools_compute.cuh"

// Compose an off-axis filtered spectrum by freezing amplitudes and shifting only the phase.

static __device__ __forceinline__ bool is_inside(int x, int y, int x_min, int x_max, int y_min, int y_max)
{
    // Simple bounding-box test; isolated for readability.
    return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
}

static __device__ __forceinline__ int wrap_index(int coordinate, int dimension)
{
    // Wrap shift coordinates in Fourier space to stay within the image lattice.
    coordinate %= dimension;
    if (coordinate < 0)
        coordinate += dimension;
    return coordinate;
}

static __global__ void kernel_off_axis_phase_mask_and_shift_frame(const cuComplex* input,
                                                                  cuComplex* scratch,
                                                                  uint width,
                                                                  uint height,
                                                                  int x_min,
                                                                  int x_max,
                                                                  int y_min,
                                                                  int y_max,
                                                                  int shift_x,
                                                                  int shift_y)
{
    // Each thread builds one complex pixel after masking and phase-only shifting.
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const size_t index = static_cast<size_t>(y) * width + x;

    const cuComplex amplitude_sample = input[index];
    const float amplitude = hypotf(amplitude_sample.x, amplitude_sample.y);
    const bool inside_current = is_inside(static_cast<int>(x), static_cast<int>(y), x_min, x_max, y_min, y_max);

    // Sample the phase from the shifted location while leaving the magnitude in place.
    const int source_x = wrap_index(static_cast<int>(x) - shift_x, static_cast<int>(width));
    const int source_y = wrap_index(static_cast<int>(y) - shift_y, static_cast<int>(height));
    const size_t source_index = static_cast<size_t>(source_y) * width + static_cast<uint>(source_x);

    const cuComplex phase_sample = input[source_index];
    const bool phase_inside = is_inside(source_x, source_y, x_min, x_max, y_min, y_max);

    float phase_unit_x = 1.0f;
    float phase_unit_y = 0.0f;

    if (phase_inside)
    {
        // Normalize the shifted complex value to extract a pure phase factor.
        const float phase_magnitude = hypotf(phase_sample.x, phase_sample.y);
        if (phase_magnitude > 0.0f)
        {
            const float inv_phase_magnitude = 1.0f / phase_magnitude;
            phase_unit_x = phase_sample.x * inv_phase_magnitude;
            phase_unit_y = phase_sample.y * inv_phase_magnitude;
        }
    }

    cuComplex result;
    if (!inside_current)
    {
        // Outside the mask we zero the phase but keep energy for continuity.
        result.x = amplitude;
        result.y = 0.0f;
    }
    else
    {
        // Recombine local magnitude with shifted phase.
        result.x = amplitude * phase_unit_x;
        result.y = amplitude * phase_unit_y;
    }

    scratch[index] = result;
}

void apply_off_axis_phase_mask_and_shift(const cuComplex* input,
                                         cuComplex* scratch,
                                         cuComplex* destination,
                                         uint width,
                                         uint height,
                                         uint frame_res,
                                         uint batch_size,
                                         int x_min,
                                         int x_max,
                                         int y_min,
                                         int y_max,
                                         int shift_x,
                                         int shift_y,
                                         const cudaStream_t stream)
{
    if (batch_size == 0 || width == 0 || height == 0)
        return;

    const uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks((width + threads_2d - 1) / threads_2d, (height + threads_2d - 1) / threads_2d);

    for (uint batch = 0; batch < batch_size; ++batch)
    {
        const size_t offset = static_cast<size_t>(batch) * frame_res;
        const cuComplex* input_frame = input + offset;
        cuComplex* destination_frame = destination + offset;

        kernel_off_axis_phase_mask_and_shift_frame<<<lblocks, lthreads, 0, stream>>>(input_frame,
                                                                                     scratch,
                                                                                     width,
                                                                                     height,
                                                                                     x_min,
                                                                                     x_max,
                                                                                     y_min,
                                                                                     y_max,
                                                                                     shift_x,
                                                                                     shift_y);

        cudaCheckError();

        cudaXMemcpyAsync(destination_frame,
                         scratch,
                         static_cast<size_t>(frame_res) * sizeof(cuComplex),
                         cudaMemcpyDeviceToDevice,
                         stream);
    }
}
