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
	}
}