/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

#include <memory>

#include <cufft.h>

namespace holovibes
{
	namespace cuda_tools
	{
		/*! \brief RAII wrapper for cufftHandle
		 */
		class CufftHandle
		{
		public:

			/*! \brief Default constructor
			 */
			CufftHandle() = default;

			/*! \brief Constructor calling plan2d
			 */
			CufftHandle(int x, int y, cufftType type);

			/*! \brief Destroy the created plan (if any)
			 */
			~CufftHandle();

			/*! \brief Destroy the created plan (if any)
			 */
			void reset();

			/*! \brief Calls plan2d
			 *
			 * Could be overloaded for plan1d and plan3d
			 */
			cufftResult plan(int x, int y, cufftType type);

			/*! \brief Calls planMany
			 */
			cufftResult planMany(int rank,
				int *n,
				int *inembed, int istride, int idist,
				int *onembed, int ostride, int odist,
				cufftType type,
				int batch);

			/*! \brief Get a reference to the underlying cufftHandle
			 */
			cufftHandle &get();

			/*! \brief Implicit cast to the underlying cufftHandle
			 */
			operator cufftHandle&();

			/*! \brief Implicit cast to a ptr to the underlying cufftHandle
			 */
			operator cufftHandle*();

		private:
			/*! \brief The cufftHandle
			 *
			 * we chose a unique_ptr to represent an possibly uninitialized one
			 */
			std::unique_ptr<cufftHandle> val_;

		};
	}
}
