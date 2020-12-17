/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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

  private:
    CusolverHandle() = delete;

    static bool initialized_;
    static cusolverDnHandle_t handle_;
};
} // namespace cuda_tools
} // namespace holovibes