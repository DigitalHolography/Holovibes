/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "cublas_handle.hh"

#include <cassert>

namespace holovibes
{
namespace cuda_tools
{
bool CublasHandle::initialized_ = false;
cublasHandle_t CublasHandle::handle_;

cublasHandle_t& CublasHandle::instance()
{
    if (!initialized_)
    {
        auto status = cublasCreate_v2(&handle_);
        assert(status == CUBLAS_STATUS_SUCCESS &&
               "Could not create cublas handle");
        initialized_ = true;
    }
    return handle_;
}
} // namespace cuda_tools
} // namespace holovibes