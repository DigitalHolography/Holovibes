/**
**	Cuda shared header file
**/
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>
# include <iostream>
# include <math.h>
# ifndef _USE_MATH_DEFINES /* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include "compute_descriptor.hh"

#define M_2PI		6.28318530718
#define THREADS_256	256
#define THREADS_128	128

typedef	unsigned int	uint;
typedef	unsigned short	ushort;
typedef	unsigned char	uchar;
typedef	cufftComplex	complex;

/* Forward declaration. */
namespace	holovibes
{
	struct	Rectangle;
	class	Queue;
	struct	UnwrappingResources;
	struct	UnwrappingResources_2d;
}

namespace	camera
{
	struct	FrameDescriptor;
}