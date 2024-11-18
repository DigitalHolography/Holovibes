#include "map.cuh"

#include "tools.hh"
#include "common.cuh"
#include "reduce.cuh"

void map_log10(float* const output, const float* const input, const size_t size, const cudaStream_t stream)
{
    static const auto log10 = [] __device__(const float input_pixel) { return log10f(input_pixel); };

    map_generic(output, input, size, log10, stream);
}

void map_divide(
    float* const output, const float* const input, const size_t size, const float value, const cudaStream_t stream)
{
    const auto divide = [value] __device__(const float input_pixel) -> float { return input_pixel / value; };

    map_generic(output, input, size, divide, stream);
}

void map_multiply(
    float* const output, const float* const input, const size_t size, const float value, const cudaStream_t stream)
{
    const auto multiply = [value] __device__(const float input_pixel) -> float { return input_pixel * value; };

    map_generic(output, input, size, multiply, stream);
}