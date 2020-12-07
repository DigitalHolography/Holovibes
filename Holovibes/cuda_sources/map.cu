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

#include "map.cuh"

#include "tools.hh"
#include "Common.cuh"
#include "reduce.cuh"

void map_log10(float* const input,
               const uint	size,
               cudaStream_t	stream)
{

    static const auto log10 = [] __device__ (const float input_pixel){ return log10f(input_pixel); };

    map_generic(input, input, size, log10, stream);
}

void map_divide(float* const input,
                const uint   size,
                const float  value,
                cudaStream_t stream)
{
    const auto divide = [value] __device__ (const float input_pixel){ return input_pixel / value; };

    map_generic(input, input, size, divide, stream);
}

void map_multiply(float* const input,
                const uint   size,
                const float  value,
                cudaStream_t stream)
{
    const auto multiply = [value] __device__ (const float input_pixel){ return input_pixel * value; };

    map_generic(input, input, size, multiply, stream);
}
