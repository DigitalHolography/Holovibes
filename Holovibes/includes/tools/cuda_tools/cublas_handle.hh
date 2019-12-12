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
	}
}