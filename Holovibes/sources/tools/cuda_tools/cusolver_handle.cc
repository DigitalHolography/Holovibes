/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "cusolver_handle.hh"

#include <cassert>

#include "common.cuh"

namespace holovibes
{
namespace cuda_tools
{
bool CusolverHandle::initialized_ = false;
cusolverDnHandle_t CusolverHandle::handle_;

cusolverDnHandle_t& CusolverHandle::instance()
{
    if (!initialized_)
    {
        auto status = cusolverDnCreate(&handle_);
        assert(status == CUSOLVER_STATUS_SUCCESS &&
               "Could not create cusolver handle");
        initialized_ = true;
    }
    return handle_;
}

void CusolverHandle::set_stream(const cudaStream_t& stream)
{
    instance();
    cusolverSafeCall(cusolverDnSetStream(handle_, stream));
}
} // namespace cuda_tools
} // namespace holovibes