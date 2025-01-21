#include "complex_utils.cuh"

/*! \brief CUDA Kernel operating a pointwise multiplication of 2 matrix.
 *
 *  \param[out] output To store the result. Same size of inputs.
 *  \param[in] input1 First matrix to multiply.
 *  \param[in] input2 Second matrix to multiply.
 *  \param[in] size Size of each matrix.
 */
__global__ void
kernel_complex_hadamard_product(cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        output[index] = input1[index] * input2[index];
}

/*! \brief CUDA Kernel operating complex conjugate.
 *
 *  \param[in out] input_output The complex matrix.
 *  \param[in] size Size of the matrix.
 */
__global__ void kernel_conjugate_complex(cuComplex* input_output, const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
        input_output[index].y = -input_output[index].y;
}

void complex_hadamard_product(
    cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_complex_hadamard_product<<<blocks, threads, 0, stream>>>(output, input1, input2, size);
    cudaCheckError();
}

void conjugate_complex(cuComplex* input_output, const uint size, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_conjugate_complex<<<blocks, threads, 0, stream>>>(input_output, size);
    cudaCheckError();
}