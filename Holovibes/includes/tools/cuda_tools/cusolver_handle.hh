/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "cusolverDn.h"

namespace holovibes
{
namespace cuda_tools
{
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
} // namespace cuda_tools
} // namespace holovibes