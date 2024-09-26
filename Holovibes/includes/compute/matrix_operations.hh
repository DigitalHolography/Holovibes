/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cusolver_handle.hh"
#include "cublas_handle.hh"
#include "common.cuh"

namespace holovibes::compute
{
/*! \brief Compute the covariance matrix of a given matrix
 *
 *  \param matrix input matrix
 *  \param width width of matrix
 *  \param height height of matrix
 *  \param cov output covariance matrix
 */
void cov_matrix(const cuComplex* matrix, int width, int height, cuComplex* cov);

/*! \brief Get the work buffer size required for eigen_values_vectors function
 *
 *  \param side length of the (triangular) matrix side
 */
int eigen_values_vectors_work_buffer_size(int side);

/*! \brief Compute the eigen values and vectors of a given triangular matrix
 *         At the end of this function: /!\ matrix == *eigen_vectors /!\
 *
 *  \param matrix input (triangular) matrix, contains eigen vectors at the end
 *  \param side length of a matrix side
 *  \param eigen_values float array that will contain the eigen values (pre
 *                      allocated)
 *  \param eigen_vectors complex matrix that will contain the eigen vectors
 *                       (should be nullptr)
 *  \param work_buffer cusolver work buffer (pre allocated)
 *  \param work_buffer_size size of the work buffer
 *  \param dev_info device information (error code)
 */
void eigen_values_vectors(cuComplex* matrix,
                          int side,
                          float* eigen_values,
                          cuComplex** eigen_vectors,
                          cuComplex* work_buffer,
                          int work_buffer_size,
                          int* dev_info);

/*! \brief Multiplies 2 complex matrices
 *
 *  \param A first input matrix
 *  \param B second input matrix
 *  \param A_height height of matrix A
 *  \param B_width width of matrix B
 *  \param A_width_B_height width of matrix A and height of matrix B
 *  \param C output matrix
 *  \param op_A operation to apply on matrix A
 *  \param op_B operation to apply on matrix B
 */
void matrix_multiply_complex(const cuComplex* A,
                             const cuComplex* B,
                             int A_height,
                             int B_width,
                             int A_width_B_height,
                             cuComplex* C,
                             cublasOperation_t op_A = CUBLAS_OP_N,
                             cublasOperation_t op_B = CUBLAS_OP_N);

/*! \brief Multiplies 2 matrices
 *
 *  \param A first input matrix
 *  \param B second input matrix
 *  \param A_height height of matrix A
 *  \param B_width width of matrix B
 *  \param A_width_B_height width of matrix A and height of matrix B
 *  \param C output matrix
 *  \param op_A operation to apply on matrix A
 *  \param op_B operation to apply on matrix B
 */
template <typename T>
void matrix_multiply(const T* A,
                     const T* B,
                     int A_height,
                     int B_width,
                     int A_width_B_height,
                     T* C,
                     cublasOperation_t op_A = CUBLAS_OP_N,
                     cublasOperation_t op_B = CUBLAS_OP_N);

/*! \brief
 *
 *  \param A first input matrix / vector
 *  \param B second input matrix / vector
 *  \param output output matrix / vector
 *  \param size size of A
 *  \param stream CUDA stream
 */
template <typename T>
void hadamard_product(const T* A, const T* B, const T* output, size_t size, const cudaStream_t stream)
} // namespace holovibes::compute
