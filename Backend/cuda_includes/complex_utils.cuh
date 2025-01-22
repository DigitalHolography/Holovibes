#pragma once

#include "cuComplex.h"
#include "cuda_runtime.h"
#include "cuda_memory.cuh"

using uint = unsigned int;

/*! \brief Multiplication operator for complex by a real. */
__host__ __device__ inline static cuComplex& operator*=(cuComplex& c, const float& r)
{
    c.x = c.x * r;
    c.y = c.y * r;
    return c;
}

/*! \brief Multiplication operator for complex by a real. */
__host__ __device__ inline static cuComplex operator*(const cuComplex& c, const float& r)
{
    cuComplex n;

    n.x = c.x * r;
    n.y = c.y * r;

    return n;
}

/*! \brief Multiplication operator for 2 complex. */
__host__ __device__ inline static cuComplex operator*(const cuComplex& c1, const cuComplex& c2)
{
    return cuCmulf(c1, c2);
}

/*! \brief Multiplication operator for 2 complex. */
__host__ __device__ inline static cuComplex& operator*=(cuComplex& c1, const cuComplex& c2)
{
    c1 = cuCmulf(c1, c2);
    return c1;
}

/*! \brief Pointwise multiplication of the pixels values of 2 complex input images.
 *
 *  \param[out] output To store the result. Same size of inputs.
 *  \param[in] input1 First matrix to multiply.
 *  \param[in] input2 Second matrix to multiply.
 *  \param[in] size Size of each matrix.
 *  \param[in] stream The CUDA stream to parallelise the computations.
 */
void complex_hadamard_product(
    cuComplex* output, const cuComplex* input1, const cuComplex* input2, const uint size, const cudaStream_t stream);

/*! \brief Convert a matrix to its complex conjugate.
 *
 *  \param[in out] input_output The complex matrix.
 *  \param[in] size Size of the matrix.
 *  \param[in] stream The CUDA stream to parallelise the computations.
 */
void conjugate_complex(cuComplex* input_output, const uint size, const cudaStream_t stream);