#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "lsci.cuh"
#include "unique_ptr.hh"
#include "test_disable_log.hh"

namespace
{
using holovibes::cuda_tools::CudaUniquePtr;

// Deliberately pool the original samples and calculate the centered variance in
// double precision, independently of the GPU's temporal-moment decomposition.
template <typename T>
std::vector<float> reference_contrast(
    const std::vector<T>& input, unsigned width, unsigned height, unsigned frames, unsigned spatial_window = 5)
{
    const int radius = static_cast<int>(spatial_window / 2);
    const size_t resolution = static_cast<size_t>(width) * height;
    std::vector<float> result(resolution);
    for (unsigned y = 0; y < height; ++y)
    {
        for (unsigned x = 0; x < width; ++x)
        {
            std::vector<double> samples;
            for (unsigned t = 0; t < frames; ++t)
                for (int row = std::max(0, static_cast<int>(y) - radius);
                     row <= std::min(static_cast<int>(height) - 1, static_cast<int>(y) + radius);
                     ++row)
                    for (int column = std::max(0, static_cast<int>(x) - radius);
                         column <= std::min(static_cast<int>(width) - 1, static_cast<int>(x) + radius);
                         ++column)
                        samples.push_back(input[t * resolution + row * width + column]);

            double mean = 0.;
            for (double sample : samples)
                mean += sample;
            mean /= samples.size();
            double variance = 0.;
            for (double sample : samples)
                variance += (sample - mean) * (sample - mean);
            variance /= samples.size();
            result[y * width + x] = mean > 0. ? static_cast<float>(std::sqrt(variance) / mean) : 0.f;
        }
    }
    return result;
}

class LsciTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0)
            GTEST_SKIP() << "LSCI numerical tests require a CUDA device";
    }

    template <typename T>
    std::vector<float> run(const std::vector<T>& input,
                           unsigned width,
                           unsigned height,
                           unsigned frames,
                           camera::PixelDepth depth,
                           unsigned batch_size = 0,
                           bool seed_previous_block = false,
                           unsigned spatial_window = 5)
    {
        const size_t resolution = static_cast<size_t>(width) * height;
        CudaUniquePtr<T> gpu_input(input.size());
        CudaUniquePtr<float2> moments(resolution);
        CudaUniquePtr<float> output(resolution);
        cudaXMemcpy(gpu_input.get(), input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice);

        if (seed_previous_block)
            lsci_accumulate(moments.get(), gpu_input.get(), resolution, frames, depth, true, 0);

        if (batch_size == 0)
            batch_size = frames;
        for (unsigned first = 0; first < frames; first += batch_size)
        {
            const unsigned count = std::min(batch_size, frames - first);
            lsci_accumulate(moments.get(),
                            gpu_input.get() + first * resolution,
                            resolution,
                            count,
                            depth,
                            first == 0,
                            0);
        }
        lsci_contrast(output.get(), moments.get(), width, height, frames, 0, spatial_window);
        cudaXStreamSynchronize(0);

        std::vector<float> result(resolution);
        cudaXMemcpy(result.data(), output.get(), resolution * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }

    static void compare(const std::vector<float>& result, const std::vector<float>& reference)
    {
        ASSERT_EQ(result.size(), reference.size());
        for (size_t pixel = 0; pixel < result.size(); ++pixel)
            EXPECT_NEAR(result[pixel], reference[pixel], 2e-5f) << "pixel " << pixel;
    }
};

TEST_F(LsciTest, PoolsRaw8BitSamplesAndClipsEveryBorder)
{
    constexpr unsigned width = 7, height = 6, frames = 16;
    std::vector<unsigned char> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[(t * height + y) * width + x] = static_cast<unsigned char>((x * 19 + y * 29 + t * 13) % 200 + 5);
    compare(run(input, width, height, frames, camera::PixelDepth::Bits8),
            reference_contrast(input, width, height, frames));
}

TEST_F(LsciTest, Handles16BitImagesSmallerThanTheSpatialKernel)
{
    constexpr unsigned width = 3, height = 2, frames = 5;
    std::vector<unsigned short> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[(t * height + y) * width + x] = static_cast<unsigned short>(500 + t * 7000 + x * 900 + y * 3500);
    compare(run(input, width, height, frames, camera::PixelDepth::Bits16),
            reference_contrast(input, width, height, frames));
}

TEST_F(LsciTest, HandlesFractionalFloatIntensitiesAndSingleRows)
{
    constexpr unsigned width = 8, height = 1, frames = 3;
    std::vector<float> input(width * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned x = 0; x < width; ++x)
            input[t * width + x] = 0.5f + x * 0.25f + t * 0.125f;
    compare(run(input, width, height, frames, camera::PixelDepth::Bits32),
            reference_contrast(input, width, height, frames));
}

TEST_F(LsciTest, IncludesSpatialVariationWhenEachPixelIsConstantInTime)
{
    constexpr unsigned width = 7, height = 5, frames = 16;
    std::vector<unsigned char> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[(t * height + y) * width + x] = static_cast<unsigned char>(10 + x * 5 + y * 3);
    const auto result = run(input, width, height, frames, camera::PixelDepth::Bits8);
    compare(result, reference_contrast(input, width, height, frames));
    EXPECT_GT(result[3 * width + 3], 0.f);
}

TEST_F(LsciTest, UsesPopulationVarianceForSpatiallyUniformTemporalVariation)
{
    constexpr unsigned width = 5, height = 5, frames = 16;
    std::vector<unsigned char> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        std::fill_n(input.begin() + t * width * height, width * height, static_cast<unsigned char>(3 * (t + 1)));
    const auto result = run(input, width, height, frames, camera::PixelDepth::Bits8);
    compare(result, reference_contrast(input, width, height, frames));
    const float expected = std::sqrt((frames * frames - 1.f) / 12.f) / ((frames + 1.f) / 2.f);
    EXPECT_NEAR(result[0], expected, 1e-6f);
}

TEST_F(LsciTest, ReturnsZeroForConstantZeroAndNonpositiveMeans)
{
    constexpr unsigned width = 3, height = 4, frames = 16;
    for (const float intensity : {0.f, 17.f, -3.f})
    {
        const std::vector<float> input(width * height * frames, intensity);
        const auto result = run(input, width, height, frames, camera::PixelDepth::Bits32);
        for (float value : result)
            EXPECT_FLOAT_EQ(value, 0.f);
    }

    // Nonpositive means can also have nonzero variance.
    const std::vector<float> input = {-1.f, 1.f, -3.f, 1.f};
    compare(run(input, 1, 1, 4, camera::PixelDepth::Bits32), reference_contrast(input, 1, 1, 4));
}

TEST_F(LsciTest, SplitBatchesAndResetPreserveTheSamePooledBlock)
{
    constexpr unsigned width = 7, height = 3, frames = 16;
    std::vector<unsigned char> input(width * height * frames);
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = static_cast<unsigned char>((index * 17) % 251);
    const auto expected = reference_contrast(input, width, height, frames);
    compare(run(input, width, height, frames, camera::PixelDepth::Bits8, 1), expected);
    compare(run(input, width, height, frames, camera::PixelDepth::Bits8, 3), expected);
    compare(run(input, width, height, frames, camera::PixelDepth::Bits8, 4, true), expected);
}

TEST_F(LsciTest, ClampsNegativeComputedVariance)
{
    CudaUniquePtr<float2> moments(size_t{1});
    CudaUniquePtr<float> output(size_t{1});
    const float2 inconsistent_moments = make_float2(2.f, 1.f);
    cudaXMemcpy(moments.get(), &inconsistent_moments, sizeof(float2), cudaMemcpyHostToDevice);
    lsci_contrast(output.get(), moments.get(), 1, 1, 1, 0);
    cudaXStreamSynchronize(0);
    float result = -1.f;
    cudaXMemcpy(&result, output.get(), sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(result, 0.f);
}

TEST_F(LsciTest, ConfigurableOddSpatialWindowsMatchIndependentPooledReference)
{
    constexpr unsigned width = 7, height = 4, frames = 5;
    std::vector<unsigned char> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[(t * height + y) * width + x] = static_cast<unsigned char>((x * 17 + y * 31 + t * 11) % 180 + 5);

    for (const unsigned side : {1u, 3u, 5u, 7u, 9u, 63u})
    {
        SCOPED_TRACE(side);
        compare(run(input, width, height, frames, camera::PixelDepth::Bits8, 2, false, side),
                reference_contrast(input, width, height, frames, side));
    }
}

TEST_F(LsciTest, SpatialWindowOneExcludesSpatialVariationAndLargerWindowsPoolIt)
{
    constexpr unsigned width = 7, height = 5, frames = 16;
    std::vector<unsigned char> input(width * height * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x)
                input[(t * height + y) * width + x] = static_cast<unsigned char>(5 + x * 9 + y * 7);

    // Every pixel is constant over time. A temporal-only estimator has zero
    // variance; pooling neighboring intensities must retain their spatial spread.
    const auto temporal_only = run(input, width, height, frames, camera::PixelDepth::Bits8, 0, false, 1);
    for (float value : temporal_only)
        EXPECT_FLOAT_EQ(value, 0.f);

    const auto narrow = run(input, width, height, frames, camera::PixelDepth::Bits8, 0, false, 3);
    const auto wide = run(input, width, height, frames, camera::PixelDepth::Bits8, 0, false, 7);
    compare(narrow, reference_contrast(input, width, height, frames, 3));
    compare(wide, reference_contrast(input, width, height, frames, 7));
    EXPECT_GT(wide[2 * width + 3] - narrow[2 * width + 3], 0.05f);
}

TEST_F(LsciTest, SpatialWindowLargerThanImageUsesOnlyAvailableSamples)
{
    // Both 63x63 windows contain exactly four samples: two pixels and two frames.
    const std::vector<unsigned char> input = {1, 3, 1, 3};
    const auto result = run(input, 2, 1, 2, camera::PixelDepth::Bits8, 0, false, 63);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 0.5f, 1e-6f);
    EXPECT_NEAR(result[1], 0.5f, 1e-6f);
    compare(result, reference_contrast(input, 2, 1, 2, 63));
}

TEST(LsciParameters, RejectsInvalidSpatialWindowSizesBeforeLaunchingCuda)
{
    for (const unsigned side : {0u, 2u, 4u, 62u, 64u, 65u, 999u})
    {
        SCOPED_TRACE(side);
        EXPECT_THROW(lsci_contrast(nullptr, nullptr, 0, 0, 0, 0, side), std::invalid_argument);
    }
    for (const unsigned side : {1u, 3u, 5u, 7u, 9u, 63u})
    {
        SCOPED_TRACE(side);
        EXPECT_NO_THROW(lsci_contrast(nullptr, nullptr, 0, 0, 0, 0, side));
    }
    EXPECT_NO_THROW(lsci_contrast(nullptr, nullptr, 0, 0, 0, 0));
}

} // namespace
