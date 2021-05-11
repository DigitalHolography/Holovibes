/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "svd.hh"

namespace holovibes
{
namespace compute
{
constexpr cuComplex alpha{1, 0};
constexpr cuComplex beta{0, 0};

void cov_matrix(const cuComplex* matrix, int width, int height, cuComplex* cov)
{

    cublasSafeCall(cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                                 CUBLAS_OP_C,
                                 CUBLAS_OP_N,
                                 height,
                                 height,
                                 width,
                                 &alpha,
                                 matrix,
                                 width,
                                 matrix,
                                 width,
                                 &beta,
                                 cov,
                                 height));
}

int eigen_values_vectors_work_buffer_size(int side)
{
    int size = 0;
    cusolverSafeCall(
        cusolverDnCheevd_bufferSize(cuda_tools::CusolverHandle::instance(),
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

void matrix_multiply(const cuComplex* A,
                     const cuComplex* B,
                     int A_height,
                     int B_width,
                     int A_width_B_height,
                     cuComplex* C,
                     cublasOperation_t op_A,
                     cublasOperation_t op_B)
{
    cublasSafeCall(cublasCgemm3m(cuda_tools::CublasHandle::instance(),
                                 op_A,
                                 op_B,
                                 A_height,
                                 B_width,
                                 A_width_B_height,
                                 &alpha,
                                 A,
                                 A_height,
                                 B,
                                 B_width,
                                 &beta,
                                 C,
                                 A_height));
}
} // namespace compute
} // namespace holovibes
