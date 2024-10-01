#include "map.cuh"

#include "tools.hh"
#include "common.cuh"
#include "reduce.cuh"

/***** Overloaded and specific map implementations *****/
void map_log10(const float* const input, float* const output, const size_t size, const cudaStream_t stream)
{
    static const auto log10 = [] __device__(const float input_pixel) { return log10f(input_pixel); };

    map_generic(input, output, size, log10, stream);
}

// It is mandatory to declare and implement these functions
// with float array parameters in order to be called from .cc

void map_divide(
    const float* const input, float* const output, const size_t size, const float value, const cudaStream_t stream)
{
    // Call templated version map divide
    map_divide<float>(input, output, size, value, stream);
}

void map_multiply(
    const float* const input, float* const output, const size_t size, const float value, const cudaStream_t stream)
{
    // Call templated version map multiply
    map_multiply<float>(input, output, size, value, stream);
}
