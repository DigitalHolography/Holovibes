/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "matrix_operations.hh"

namespace holovibes::compute
{
template <typename T>
void matrix_multiply(const T* A,
                     const T* B,
                     int A_height,
                     int B_width,
                     int A_width_B_height,
                     T* C,
                     cublasOperation_t op_A,
                     cublasOperation_t op_B)
{
    cublasSafeCall(cublasSgemm(cuda_tools::CublasHandle::instance(),
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
} // namespace holovibes::compute