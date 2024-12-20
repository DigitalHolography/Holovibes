/*! \file
 *
 * \brief declaration of the CublasHandle class
 */
#pragma once

#include <cublas_v2.h>

namespace holovibes::cuda_tools
{
/*! \class CublasHandle
 *
 * \brief Singleton class that manages the cublas handle
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
