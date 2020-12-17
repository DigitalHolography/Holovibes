/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <cublas_v2.h>

namespace holovibes
{
namespace cuda_tools
{
class CublasHandle
{
  public:
    static cublasHandle_t& instance();

  private:
    CublasHandle() = delete;

    static bool initialized_;
    static cublasHandle_t handle_;
};
} // namespace cuda_tools
} // namespace holovibes