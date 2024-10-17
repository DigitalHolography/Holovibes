#include "reduce.cuh"

static constexpr cudaStream_t stream = 0;

void test_gpu_reduce_add(double* const output, const float* const input, const uint size)
{
    reduce_add(output, input, size, stream);
}

void test_gpu_reduce_min(double* const output, const double* const input, const uint size)
{
    reduce_min(output, input, size, stream);
}

void test_gpu_reduce_max(int* const output, const int* const input, const uint size)
{
    reduce_max(output, input, size, stream);
}

void test_gpu_reduce_max(float* const output, const float* const input, const uint size)
{
    reduce_max(output, input, size, stream);
}
