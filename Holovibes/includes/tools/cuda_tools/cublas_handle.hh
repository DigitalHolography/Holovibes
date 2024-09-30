/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include <cublas_v2.h>

namespace holovibes::cuda_tools
{
/*! \class CublasHandle
 *
 * \brief #TODO Add a description for this class
 */
class CublasHandle
{
  public:
    static cublasHandle_t& instance();
    static void set_stream(const cudaStream_t& stream);

  private:
    CublasHandle() = delete;

    static bool initialized_;
    static cublasHandle_t handle_;
};
} // namespace holovibes::cuda_tools
