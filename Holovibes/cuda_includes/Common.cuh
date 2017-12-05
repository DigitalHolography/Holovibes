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

# include <device_launch_parameters.h>
# include <string>
# include <exception>

# include "tools.cuh"
# include "queue.hh"
# include "compute_descriptor.hh"
# include "custom_exception.hh"

#define M_2PI		6.28318530717959f
#define THREADS_256	256
#define THREADS_128	128



#define cudaCheckError()                                                     \
{                                                                            \
	auto e = cudaGetLastError();                                             \
	if (e != cudaSuccess)                                                    \
	{                                                                        \
		std::string error = "Cuda failure in ";                              \
		error += __FILE__;                                                   \
		error += " at line ";                                                \
		error += std::to_string(__LINE__);                                   \
		error += ": ";                                                       \
		error += cudaGetErrorString(e);                                      \
		throw holovibes::CustomException(error, holovibes::fail_cudaLaunch); \
	}												                         \
}