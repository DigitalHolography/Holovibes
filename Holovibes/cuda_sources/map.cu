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
#include "common.cuh"
#include "reduce.cuh"

/***** Overloaded and specific map implementations *****/
void map_log10(const float* const input,
               float* const output,
               const size_t	size,
               const cudaStream_t	stream)
{
    static const auto log10 = [] __device__ (const float input_pixel){ return log10f(input_pixel); };

    map_generic(input, output, size, log10, stream);
}

// It is mandatory to declare and implement these functions
// with float array parameters in order to be called from .cc

void map_divide(const float* const input,
                float* const output,
                const size_t size,
                const float value,
                const cudaStream_t stream)
{
    // Call templated version map divide
    map_divide<float>(input, output, size, value, stream);
}

void map_multiply(const float* const input,
                  float* const output,
                  const size_t size,
                  const float value,
                  const cudaStream_t stream)
{
    // Call templated version map multiply
    map_multiply<float>(input, output, size, value, stream);
}