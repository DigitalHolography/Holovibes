#include "reduce.cuh"

static constexpr cudaStream_t stream = 0;

void test_gpu_reduce_add(const float* const input, double* const result, const uint size)
{
    reduce_add(input, result, size, stream);
}

void test_gpu_reduce_min(const double* const input, double* const result, const uint size)
{
    reduce_min(input, result, size, stream);
}

void test_gpu_reduce_max(const int* const input, int* const result, const uint size)
{
    reduce_max(input, result, size, stream);
}

void test_gpu_reduce_max(const float* const input, float* const result, const uint size)
{
    reduce_max(input, result, size, stream);
}
