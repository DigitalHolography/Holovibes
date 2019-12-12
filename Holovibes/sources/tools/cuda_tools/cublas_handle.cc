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
				assert(status == CUBLAS_STATUS_SUCCESS && "Could not create cublas handle");
				initialized_ = true;
			}
			return handle_;
		}
	}
}