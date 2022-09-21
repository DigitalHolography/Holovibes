#include "cublas_handle.hh"

#include <cassert>

#include "common.cuh"

namespace holovibes
{
namespace cuda_tools
{
bool CublasHandle::initialized_ = false;
cublasHandle_t CublasHandle::handle_;
cudaStream_t stream_;

cublasHandle_t& CublasHandle::instance()
{
    if (!initialized_)
    {
        [[maybe_unused]] auto status = cublasCreate_v2(&handle_);
        CHECK(status == CUBLAS_STATUS_SUCCESS, "Could not create cublas handle");
        initialized_ = true;
    }
    return handle_;
}

void CublasHandle::set_stream(const cudaStream_t& stream)
{
    instance();
    cublasSafeCall(cublasSetStream(handle_, stream));
}
} // namespace cuda_tools
} // namespace holovibes
