/*! \file
 *
 * \brief Declaration of the CusolverHandle class
 */
#pragma once

#include "cusolverDn.h"

namespace holovibes::cuda_tools
{
/*! \class CusolverHandle
 *
 * \brief Singleton class that manages the cusolver handle
 */
class CusolverHandle
{
  public:
    static cusolverDnHandle_t& instance();
    static void set_stream(const cudaStream_t& stream);

  private:
    CusolverHandle() = delete;

    static bool initialized_;
    static cusolverDnHandle_t handle_;
};
} // namespace holovibes::cuda_tools
