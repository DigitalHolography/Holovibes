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

# include <cuda_runtime.h>
# include <cufft.h>
# include <iostream>
# include <device_launch_parameters.h>
# include "compute_descriptor.hh"

# ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
# endif
# include <cmath>
# include <math.h>
#define M_2PI		6.28318530717959f
#define THREADS_256	256
#define THREADS_128	128

typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
typedef	cufftComplex	complex;

/* Forward declaration. */
namespace	holovibes
{
	class	Queue;
	struct	UnwrappingResources;
	struct	UnwrappingResources_2d;
}

namespace	camera
{
	struct	FrameDescriptor;
}