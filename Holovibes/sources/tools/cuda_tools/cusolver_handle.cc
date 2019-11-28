#include "cusolver_handle.hh"

#include <cassert>

namespace cuda_tools
{
	bool CusolverHandle::initialized_ = false;
	cusolverDnHandle_t CusolverHandle::handle_;

	cusolverDnHandle_t& CusolverHandle::instance()
	{
		if (!initialized_)
		{
			auto status = cusolverDnCreate(&handle_);
			assert(status == CUSOLVER_STATUS_SUCCESS && "Could not create cusolver handle");
			initialized_ = true;
		}
		return handle_;
	}
}