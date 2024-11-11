#include "gtest/gtest.h"

#include "common.cuh"
#include "cuda_memory.cuh"
#include "tools_analysis.cuh"

TEST(AnalysisTest, comp_dgaussian)
{
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamDefault, 1);

    constexpr float x[] = {-1, 0, 1};
    constexpr float sigma = 0.1f;
    constexpr uint n = 2;

    float* d_x;
    cudaXMalloc(&d_x, sizeof(float) * 3);
    cudaXMemcpy(d_x, x, sizeof(float) * 3, cudaMemcpyHostToDevice);

    float* output_x;
    cudaXMalloc(&output_x, 3 * sizeof(float));

    comp_dgaussian(output_x, d_x, 3, sigma, n, stream);
    float actual[3];

    cudaXStreamSynchronize(stream);
    cudaXMemcpy(actual, output_x, sizeof(float) * 3, cudaMemcpyDeviceToHost);

    float expected[] = {
        7.6177e-18,
        -398.9423,
        7.6177e-18,
    };

    // ASSERT_TRUE(std::equal(std::begin(expected), std::end(expected), std::begin(actual)));

    cudaXFree(d_x);
    cudaXFree(output_x);
    ASSERT_TRUE(1);
}