#include "cusolver_handle.hh"

#include <cassert>

#include "common.cuh"

namespace holovibes::cuda_tools
{
bool CusolverHandle::initialized_ = false;
cusolverDnHandle_t CusolverHandle::handle_;

cusolverDnHandle_t& CusolverHandle::instance()
{
    if (!initialized_)
    {
        [[maybe_unused]] auto status = cusolverDnCreate(&handle_);
        CHECK(status == CUSOLVER_STATUS_SUCCESS, "Could not create cusolver handle");
        initialized_ = true;
    }
    return handle_;
}

void CusolverHandle::set_stream(const cudaStream_t& stream)
{
    instance();
    cusolverSafeCall(cusolverDnSetStream(handle_, stream));
}
} // namespace holovibes::cuda_tools
