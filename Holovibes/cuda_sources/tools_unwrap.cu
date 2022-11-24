#include "tools_unwrap.cuh"

#include "cuda_memory.cuh"
#include "frame_desc.hh"

using namespace holovibes;

__global__ void kernel_extract_angle(const cuComplex* input, float* output, const size_t size)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        // We use std::atan2 in order to obtain results in [-pi; pi].
        output[index] = std::atan2(input[index].y, input[index].x);
    }
}

__global__ void kernel_unwrap(const float* pred, const float* cur, float* output, const size_t size)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        const float local_diff = cur[index] - pred[index];
        // Unwrapping //
        float local_adjust;
        if (local_diff > M_PI)
            local_adjust = -M_2PI;
        else if (local_diff < -M_PI)
            local_adjust = M_2PI;
        else
            local_adjust = 0.f;
        // Cumulating each angle with its correction
        output[index] = cur[index] + local_adjust;
    }
}

__global__ void kernel_compute_angle_mult(const cuComplex* pred, const cuComplex* cur, float* output, const size_t size)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        cuComplex conj_prod;
        conj_prod = cur[index];

        conj_prod.x *= pred[index].x;
        conj_prod.x += cur[index].y * pred[index].y;

        conj_prod.y *= pred[index].x;
        conj_prod.y -= cur[index].x * pred[index].y;

        output[index] = std::atan2(conj_prod.y, conj_prod.x);
    }
}

__global__ void kernel_compute_angle_diff(const cuComplex* pred, const cuComplex* cur, float* output, const size_t size)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size)
    {
        cuComplex diff = cur[index];
        diff.x -= pred[index].x;
        diff.y -= pred[index].y;
        output[index] = std::atan2(diff.y, diff.x);
    }
}

__global__ void
kernel_correct_angles(float* data, const float* corrections, const size_t image_size, const size_t history_size)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t size = history_size * image_size;
    if (index < image_size)
        for (auto correction_idx = index; correction_idx < size; correction_idx += image_size)
            data[index] += corrections[correction_idx];
}

__global__ void kernel_init_unwrap_2d(
    const uint width, const uint height, const uint frame_res, const float* input, float* fx, float* fy, cuComplex* z)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = j * blockDim.x * gridDim.x + i;

    if (index < frame_res)
    {
        fx[index] = (i - static_cast<float>(lrintf(static_cast<float>(width >> 1))));
        fy[index] = (j - static_cast<float>(lrintf(static_cast<float>(height >> 1))));

        /*z init*/
        z[index].x = cosf(input[index]);
        z[index].y = sinf(input[index]);
    }
}

__global__ void kernel_multiply_complexes_by_floats_(
    const float* input1, const float* input2, cuComplex* output1, cuComplex* output2, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output1[index].x *= input1[index];
        output1[index].y *= input1[index];
        output2[index].x *= input2[index];
        output2[index].y *= input2[index];
    }
}

__global__ void kernel_multiply_complexes_by_single_complex(cuComplex* output1,
                                                            cuComplex* output2,
                                                            const cuComplex input,
                                                            const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        const cuComplex cpy_o1 = output1[index];
        const cuComplex cpy_o2 = output2[index];

        output1[index].x = cpy_o1.x * input.x - cpy_o1.y * input.y;
        output1[index].y = cpy_o1.x * input.y + cpy_o1.y * input.x;
        output2[index].x = cpy_o2.x * input.x - cpy_o2.y * input.y;
        output2[index].y = cpy_o2.x * input.y + cpy_o2.y * input.x;
    }
}

__global__ void kernel_multiply_complex_by_single_complex(cuComplex* output, const cuComplex input, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        const cuComplex cpy_o1 = output[index];

        output[index].x = cpy_o1.x * input.x - cpy_o1.y * input.y;
        output[index].y = cpy_o1.x * input.y + cpy_o1.y * input.x;
    }
}

__global__ void kernel_conjugate_complex(cuComplex* output, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index].y = -output[index].y;
    }
}

__global__ void kernel_multiply_complex_frames_by_complex_frame(cuComplex* output1,
                                                                cuComplex* output2,
                                                                const cuComplex* input,
                                                                const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        const cuComplex cpy_o1 = output1[index];
        const cuComplex cpy_o2 = output2[index];

        output1[index].x = cpy_o1.x * input[index].x - cpy_o1.y * input[index].y;
        output1[index].y = cpy_o1.x * input[index].y + cpy_o1.y * input[index].x;
        output2[index].x = cpy_o2.x * input[index].x - cpy_o2.y * input[index].y;
        output2[index].y = cpy_o2.x * input[index].y + cpy_o2.y * input[index].x;
    }
}

__global__ void
kernel_norm_ratio(const float* input1, const float* input2, cuComplex* output1, cuComplex* output2, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        const float norm = input1[index] * input1[index] + input2[index] * input2[index];

        if (norm != 0)
        {
            const float coeff_x = input1[index] / norm;
            const float coeff_y = input2[index] / norm;

            output1[index].x = output1[index].x * coeff_x;
            output1[index].y = output1[index].y * coeff_x;
            output2[index].x = output2[index].x * coeff_y;
            output2[index].y = output2[index].y * coeff_y;
        }
        else
        {
            output1[index].x = 0;
            output1[index].y = 0;
            output2[index].x = 0;
            output2[index].y = 0;
        }
    }
}

__global__ void kernel_add_complex_frames(cuComplex* output, const cuComplex* input, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        output[index].x += input[index].x;
        output[index].y += input[index].y;
    }
}

__global__ void kernel_unwrap2d_last_step(float* output, const cuComplex* input, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        output[index] = input[index].y / -M_2PI;
}

void phase_increase(const cuComplex* cur,
                    UnwrappingResources* resources,
                    const size_t image_size,
                    const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(image_size, threads);
    static bool first_time = true;
    if (first_time)
    {
        cudaXMemcpyAsync(resources->gpu_predecessor_,
                         cur,
                         sizeof(cuComplex) * image_size,
                         cudaMemcpyDeviceToDevice,
                         stream);
        first_time = false;
    }

    // Compute the newest phase image, not unwrapped yet
    kernel_compute_angle_mult<<<blocks, threads, 0, stream>>>(resources->gpu_predecessor_,
                                                              cur,
                                                              resources->gpu_angle_current_,
                                                              image_size);
    cudaCheckError();
    // Updating predecessor (complex image) for the next iteration
    cudaXMemcpyAsync(resources->gpu_predecessor_,
                     cur,
                     sizeof(cuComplex) * image_size,
                     cudaMemcpyDeviceToDevice,
                     stream);

    /* Copying in order to later enqueue the (not summed up with values
     * in gpu_unwrap_buffer_) phase image. */
    cudaXMemcpyAsync(resources->gpu_angle_copy_,
                     resources->gpu_angle_current_,
                     sizeof(float) * image_size,
                     cudaMemcpyDeviceToDevice,
                     stream);

    // Applying history on the latest phase image
    kernel_correct_angles<<<blocks, threads, 0, stream>>>(resources->gpu_angle_current_,
                                                          resources->gpu_unwrap_buffer_,
                                                          image_size,
                                                          resources->size_);
    cudaCheckError();

    /* Store the new phase image in the next buffer position.
     * The buffer is handled as a circular buffer. */
    float* next_unwrap = resources->gpu_unwrap_buffer_ + image_size * resources->next_index_;
    cudaXMemcpyAsync(next_unwrap,
                     resources->gpu_angle_copy_,
                     sizeof(float) * image_size,
                     cudaMemcpyDeviceToDevice,
                     stream);
    if (resources->size_ < resources->capacity_)
        ++resources->size_;
    resources->next_index_ = (resources->next_index_ + 1) % resources->capacity_;
}

void unwrap_2d(float* input,
               const cufftHandle plan2d,
               UnwrappingResources_2d* res,
               const FrameDescriptor& fd,
               float* output,
               const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);
    kernel_init_unwrap_2d<<<lblocks, lthreads, 0, stream>>>(fd.width,
                                                            fd.height,
                                                            fd.get_frame_res(),
                                                            input,
                                                            res->gpu_fx_,
                                                            res->gpu_fy_,
                                                            res->gpu_z_);
    cudaCheckError();
    ushort middlex = fd.width >> 1;
    ushort middley = fd.height >> 1;
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);
    circ_shift_float<<<blocks, threads, 0, stream>>>(res->gpu_fx_,
                                                     res->gpu_shift_fx_,
                                                     1,
                                                     middlex,
                                                     middley,
                                                     fd.width,
                                                     fd.height,
                                                     fd.get_frame_res());
    cudaCheckError();
    circ_shift_float<<<blocks, threads, 0, stream>>>(res->gpu_fy_,
                                                     res->gpu_shift_fy_,
                                                     1,
                                                     middlex,
                                                     middley,
                                                     fd.width,
                                                     fd.height,
                                                     fd.get_frame_res());
    cudaCheckError();
    gradient_unwrap_2d(plan2d, res, fd, stream);
    eq_unwrap_2d(plan2d, res, fd, stream);
    phi_unwrap_2d(plan2d, res, fd, output, stream);
}

void gradient_unwrap_2d(const cufftHandle plan2d,
                        UnwrappingResources_2d* res,
                        const FrameDescriptor& fd,
                        const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);
    cuComplex single_complex = make_cuComplex(0.f, static_cast<float>(M_2PI));

    cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
    cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
    kernel_multiply_complexes_by_floats_<<<blocks, threads, 0, stream>>>(res->gpu_shift_fx_,
                                                                         res->gpu_shift_fy_,
                                                                         res->gpu_grad_eq_x_,
                                                                         res->gpu_grad_eq_y_,
                                                                         fd.get_frame_res());
    cudaCheckError();
    cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
    cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_INVERSE);
    kernel_multiply_complexes_by_single_complex<<<blocks, threads, 0, stream>>>(res->gpu_grad_eq_x_,
                                                                                res->gpu_grad_eq_y_,
                                                                                single_complex,
                                                                                fd.get_frame_res());
    cudaCheckError();
}

void eq_unwrap_2d(const cufftHandle plan2d,
                  UnwrappingResources_2d* res,
                  const FrameDescriptor& fd,
                  const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);
    cuComplex single_complex = make_cuComplex(0, 1);

    kernel_multiply_complex_by_single_complex<<<blocks, threads, 0, stream>>>(res->gpu_z_,
                                                                              single_complex,
                                                                              fd.get_frame_res());
    cudaCheckError();
    kernel_conjugate_complex<<<blocks, threads, 0, stream>>>(res->gpu_z_, fd.get_frame_res());
    cudaCheckError();
    kernel_multiply_complex_frames_by_complex_frame<<<blocks, threads, 0, stream>>>(res->gpu_grad_eq_x_,
                                                                                    res->gpu_grad_eq_y_,
                                                                                    res->gpu_z_,
                                                                                    fd.get_frame_res());
    cudaCheckError();
    cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
    cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
    kernel_norm_ratio<<<blocks, threads, 0, stream>>>(res->gpu_shift_fx_,
                                                      res->gpu_shift_fy_,
                                                      res->gpu_grad_eq_x_,
                                                      res->gpu_grad_eq_y_,
                                                      fd.get_frame_res());
    cudaCheckError();
}

void phi_unwrap_2d(const cufftHandle plan2d,
                   UnwrappingResources_2d* res,
                   const FrameDescriptor& fd,
                   float* output,
                   const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(res->image_resolution_, threads);

    kernel_add_complex_frames<<<blocks, threads, 0, stream>>>(res->gpu_grad_eq_x_,
                                                              res->gpu_grad_eq_y_,
                                                              fd.get_frame_res());
    cudaCheckError();
    cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
    kernel_unwrap2d_last_step<<<blocks, threads, 0, stream>>>(output, res->gpu_grad_eq_x_, fd.get_frame_res());
    cudaCheckError();
}
