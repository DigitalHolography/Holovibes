#include "matrix_operations.hh"
#include "logger.hh"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace holovibes::compute
{
void cov_matrix(const cuComplex* matrix, int width, int height, cuComplex* cov)
{

    cublasSafeCall(cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                                 CUBLAS_OP_C,
                                 CUBLAS_OP_N,
                                 height,
                                 height,
                                 width,
                                 &alpha_complex,
                                 matrix,
                                 width,
                                 matrix,
                                 width,
                                 &beta_complex,
                                 cov,
                                 height));
}

int eigen_values_vectors_work_buffer_size(int side)
{
    // LOG-USELESS LOG_FUNC(side);

    int size = 0;
    cusolverSafeCall(cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(),
                                                 CUSOLVER_EIG_MODE_VECTOR,
                                                 CUBLAS_FILL_MODE_LOWER,
                                                 side,
                                                 nullptr,
                                                 side,
                                                 nullptr,
                                                 &size));
    return size;
}

void eigen_values_vectors(cuComplex* matrix,
                          int side,
                          float* eigen_values,
                          cuComplex** eigen_vectors,
                          cuComplex* work_buffer,
                          int work_buffer_size,
                          int* dev_info)
{
    // LOG-USELESS LOG_FUNC();

    *eigen_vectors = matrix;
    cusolverSafeCall(cusolverDnCheevd(cuda_tools::CusolverHandle::instance(),
                                      CUSOLVER_EIG_MODE_VECTOR,
                                      CUBLAS_FILL_MODE_LOWER,
                                      side,
                                      matrix,
                                      side,
                                      eigen_values,
                                      work_buffer,
                                      work_buffer_size,
                                      dev_info));
}

void matrix_argmax(const float* matrix, const short width, const short height, int& max_index, int& x, int& y)
{
    cublasSafeCall(cublasIsamax(cuda_tools::CublasHandle::instance(), width * height, matrix, 1, &max_index));
    max_index--; // Cublas start couting index from 1.

    // Convert the linear index to (x, y) coordinates
    x = max_index % width; // Column
    y = max_index / width; // Row
}

void matrix_multiply_complex(const cuComplex* A,
                             const cuComplex* B,
                             int A_height,
                             int B_width,
                             int A_width_B_height,
                             cuComplex* C,
                             cublasOperation_t op_A,
                             cublasOperation_t op_B)
{
    // LOG-USELESS LOG_FUNC();

    cublasSafeCall(cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                                 op_A,
                                 op_B,
                                 A_height,
                                 B_width,
                                 A_width_B_height,
                                 &alpha_complex,
                                 A,
                                 A_height,
                                 B,
                                 B_width,
                                 &beta_complex,
                                 C,
                                 A_height));
}
} // namespace holovibes::compute
