#include "apply_mask.cuh"

#include "hardware_limits.hh"
#include "common.cuh"

__host__ __device__ static cuComplex& operator*=(cuComplex& c, const float& r)
{
    c.x = c.x * r;
    c.y = c.y * r;
    return c;
}

__host__ __device__ static cuComplex operator*(const cuComplex& c, const float& r)
{
    cuComplex n;

    n.x = c.x * r;
    n.y = c.y * r;

    return n;
}

__host__ __device__ static cuComplex operator*(const cuComplex& c1, const cuComplex& c2) { return cuCmulf(c1, c2); }

__host__ __device__ static cuComplex& operator*=(cuComplex& c1, const cuComplex& c2)
{
    c1 = cuCmulf(c1, c2);
    return c1;
}

template <typename T, typename M>
__global__ static void kernel_apply_mask(T* in_out, const M* mask, const size_t size, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            in_out[(size * i) + index] *= mask[index];
        }
    }
}

template <typename T, typename M>
__global__ static void
kernel_apply_mask(const T* input, const M* mask, T* output, const size_t size, const uint batch_size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        for (uint i = 0; i < batch_size; ++i)
        {
            output[(size * i) + index] = input[(size * i) + index] * mask[index];
        }
    }
}

template <typename T, typename M>
static void
apply_mask_caller(T* in_out, const M* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_apply_mask<T, M><<<blocks, threads, 0, stream>>>(in_out, mask, size, batch_size);
    cudaCheckError();
}

template <typename T, typename M>
static void apply_mask_caller(
    const T* input, const M* mask, T* output, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);

    kernel_apply_mask<T, M><<<blocks, threads, 0, stream>>>(input, mask, output, size, batch_size);
    cudaCheckError();
}

void apply_mask(
    cuComplex* in_out, const cuComplex* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, cuComplex>(in_out, mask, size, batch_size, stream);
}

void apply_mask(
    cuComplex* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, float>(in_out, mask, size, batch_size, stream);
}

void apply_mask(float* in_out, const float* mask, const size_t size, const uint batch_size, const cudaStream_t stream)
{
    apply_mask_caller<float, float>(in_out, mask, size, batch_size, stream);
}

void apply_mask(const cuComplex* input,
                const cuComplex* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, cuComplex>(input, mask, output, size, batch_size, stream);
}

void apply_mask(const cuComplex* input,
                const float* mask,
                cuComplex* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<cuComplex, float>(input, mask, output, size, batch_size, stream);
}

void apply_mask(const float* input,
                const float* mask,
                float* output,
                const size_t size,
                const uint batch_size,
                const cudaStream_t stream)
{
    apply_mask_caller<float, float>(input, mask, output, size, batch_size, stream);
}
