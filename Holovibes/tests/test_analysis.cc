#include "gtest/gtest.h"

#include "cuda_memory.cuh"
#include "vesselness_filter.cuh"
#include "batch_input_queue.hh"

#include <thread>

TEST(AnalysisTest, comp_dgaussian_Gqy)
{
    constexpr float y[] = {-1, 0, 1};
    constexpr float sigma = 0.1f;
    constexpr uint n = 2;

    float* actual = comp_dgaussian(x, sigma, n, 3);
    float[] expected = {
        7.6177e-18,
        -398.9423,
        7.6177e-18,
    };

    ASSERT_TRUE(std::equal(std::begin(expected), std::end(expected), std::begin(actual)));
}

TEST(AnalysisTest, comp_dgaussian_Gqy)
{
    constexpr float y[] = {-1, 0, 1};
    constexpr float sigma = 0.1f;
    constexpr uint n = 0;

    float* actual = comp_dgaussian(x, sigma, n, 3);
    float[] expected = {
        7.6946e-22,
        3.9894,
        7.6946e-22,
    };

    ASSERT_TRUE(std::equal(std::begin(expected), std::end(expected), std::begin(actual)));
}
