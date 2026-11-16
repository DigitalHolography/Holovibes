#include "gtest/gtest.h"
#include "lsci_pca.hh"
#include "test_disable_log.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

namespace
{
struct Reference
{
    cv::Mat original, reconstructed;
    unsigned components = 0;
};

struct PatchReference
{
    unsigned x, y, width, height, components;
};

struct LocalReference
{
    Reference images;
    std::vector<PatchReference> patches;
};

// Deliberately reconstruct every time sample in double precision on the CPU,
// independently of the GPU score-moment identity used by the implementation.
template <typename T>
Reference reconstruct(const std::vector<T>& input, unsigned pixels, unsigned frames)
{
    Reference r;
    r.original.create(pixels, frames, CV_64F);
    cv::Mat centered(pixels, frames, CV_64F), means(1, frames, CV_64F);
    for (unsigned t = 0; t < frames; ++t)
    {
        double sum = 0.;
        for (unsigned p = 0; p < pixels; ++p)
            sum += (r.original.at<double>(p, t) = input[t*pixels+p]);
        means.at<double>(t) = sum / pixels;
        for (unsigned p = 0; p < pixels; ++p)
            centered.at<double>(p, t) = r.original.at<double>(p, t) - sum/pixels;
    }
    cv::Mat gram = centered.t() * centered;
    cv::Mat values, vectors;
    EXPECT_TRUE(cv::eigen(gram, values, vectors));
    const double total = cv::sum(values)[0];
    if (total > 0.)
        for (unsigned i = 0; i < frames; ++i)
            r.components += values.at<double>(i) >= total/frames;
    r.reconstructed = cv::Mat::zeros(pixels, frames, CV_64F);
    if (r.components)
    {
        cv::Mat basis = vectors.rowRange(0, r.components).t();
        r.reconstructed = centered * basis * basis.t();
    }
    for (unsigned p = 0; p < pixels; ++p)
        for (unsigned t = 0; t < frames; ++t)
            r.reconstructed.at<double>(p, t) += means.at<double>(t);
    return r;
}

// Reconstruct each patch independently, then assemble the full movie before
// calculating contrast. Spatial contrast windows may cross PCA patch borders.
template <typename T>
LocalReference reconstruct_local(const std::vector<T>& input, unsigned width, unsigned height,
                                 unsigned frames, unsigned patch_size)
{
    LocalReference r;
    const unsigned pixels = width * height;
    r.images.original.create(pixels, frames, CV_64F);
    r.images.reconstructed.create(pixels, frames, CV_64F);
    for (unsigned t = 0; t < frames; ++t)
        for (unsigned p = 0; p < pixels; ++p)
            r.images.original.at<double>(p, t) = input[t*pixels+p];
    for (unsigned y = 0; y < height; y += patch_size)
        for (unsigned x = 0; x < width; x += patch_size)
        {
            const unsigned w = std::min(patch_size, width-x), h = std::min(patch_size, height-y);
            std::vector<T> patch(w*h*frames);
            for (unsigned t = 0; t < frames; ++t)
                for (unsigned row = 0; row < h; ++row)
                    for (unsigned column = 0; column < w; ++column)
                        patch[t*w*h+row*w+column] = input[t*pixels+(y+row)*width+x+column];
            const auto local = reconstruct(patch, w*h, frames);
            r.patches.push_back({x, y, w, h, local.components});
            for (unsigned t = 0; t < frames; ++t)
                for (unsigned row = 0; row < h; ++row)
                    for (unsigned column = 0; column < w; ++column)
                        r.images.reconstructed.at<double>((y+row)*width+x+column, t) =
                            local.reconstructed.at<double>(row*w+column, t);
        }
    return r;
}

double pooled_k(const cv::Mat& stack, unsigned x, unsigned y, unsigned w, unsigned h, unsigned side)
{
    std::vector<double> samples;
    const int radius = side/2;
    for (int row = std::max(0, static_cast<int>(y)-radius); row <= std::min<int>(h-1, y+radius); ++row)
        for (int column = std::max(0, static_cast<int>(x)-radius); column <= std::min<int>(w-1, x+radius); ++column)
            for (int t = 0; t < stack.cols; ++t)
                samples.push_back(stack.at<double>(row*w+column, t));
    double mean = 0., variance = 0.;
    for (double sample : samples)
        mean += sample;
    mean /= samples.size();
    for (double sample : samples)
        variance += (sample-mean)*(sample-mean);
    return mean > 0. ? std::sqrt(variance/samples.size())/mean : 0.;
}

class LsciPcaTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        int count;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
            GTEST_SKIP() << "PCA-LSCI numerical tests require CUDA";
    }

    template <typename T>
    void compare(const std::vector<T>& input, unsigned w, unsigned h, unsigned n, unsigned side,
                 camera::PixelDepth depth, unsigned patch_size = 0)
    {
        holovibes::compute::LsciPca processor(w, h, n, side, patch_size);
        const auto local = patch_size ? reconstruct_local(input, w, h, n, patch_size) : LocalReference{};
        const auto reference = patch_size ? local.images : reconstruct(input, w*h, n);
        // Reuse the same processor to check that consecutive blocks do not mix.
        for (unsigned block = 0; block < 2; ++block)
        {
            for (unsigned t = 0; t < n; ++t)
                processor.add_frame(input.data()+t*w*h, depth);
            const auto output = processor.process();
            ASSERT_EQ(output.static_components, reference.components);
            ASSERT_EQ(output.patches.size(), local.patches.size());
            if (patch_size)
            {
                EXPECT_TRUE(output.eigenvalues.empty());
                for (size_t p = 0; p < local.patches.size(); ++p)
                {
                    const auto& actual = output.patches[p];
                    const auto& expected = local.patches[p];
                    EXPECT_EQ(actual.x, expected.x);
                    EXPECT_EQ(actual.y, expected.y);
                    EXPECT_EQ(actual.width, expected.width);
                    EXPECT_EQ(actual.height, expected.height);
                    EXPECT_EQ(actual.static_components, expected.components);
                    EXPECT_EQ(actual.eigenvalues.size(), n);
                }
            }
            for (unsigned y = 0; y < h; ++y)
                for (unsigned x = 0; x < w; ++x)
                {
                    const double original = pooled_k(reference.original, x, y, w, h, side);
                    const double stat = pooled_k(reference.reconstructed, x, y, w, h, side);
                    EXPECT_NEAR(output.original[y*w+x], original, 2e-5);
                    EXPECT_NEAR(output.static_component[y*w+x], stat, 5e-4);
                    EXPECT_NEAR(output.dynamic[y*w+x], original-stat, 5e-4);
                }
        }
    }
};

TEST_F(LsciPcaTest, MatchesFullCpuReconstructionWithMultipleStaticComponentsAndClippedBorders)
{
    constexpr unsigned w = 13, h = 11, n = 8;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        for (unsigned p = 0; p < w*h; ++p)
            input[t*w*h+p] = static_cast<unsigned char>(120 + (p%2 ? 30 : -30) +
                (p%3 ? 18 : -18)*std::cos(6.283185307179586*t/n) + static_cast<int>((p*13+t*7)%11)-5);
    const auto reference = reconstruct(input, w*h, n);
    ASSERT_GE(reference.components, 2u);
    compare(input, w, h, n, 7, camera::PixelDepth::Bits8);
}

TEST_F(LsciPcaTest, Supports16BitAndFloatIntensities)
{
    constexpr unsigned w = 9, h = 7, n = 6;
    std::vector<unsigned short> shorts(w*h*n);
    std::vector<float> floats(w*h*n);
    for (unsigned i = 0; i < shorts.size(); ++i)
    {
        shorts[i] = static_cast<unsigned short>(1000+(i*977)%50000);
        floats[i] = shorts[i]*.001f;
    }
    compare(shorts, w, h, n, 3, camera::PixelDepth::Bits16);
    compare(floats, w, h, n, 3, camera::PixelDepth::Bits32);
}

TEST_F(LsciPcaTest, StationaryTextureAndZeroSignalHaveNoDynamicContrast)
{
    constexpr unsigned w = 9, h = 7, n = 8;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        for (unsigned p = 0; p < w*h; ++p)
            input[t*w*h+p] = static_cast<unsigned char>(10+(p*17)%180);
    compare(input, w, h, n, 3, camera::PixelDepth::Bits8);
    std::fill(input.begin(), input.end(), 0);
    compare(input, w, h, n, 3, camera::PixelDepth::Bits8);
}

TEST_F(LsciPcaTest, RejectsPartialBlocksAndInvalidIntensities)
{
    holovibes::compute::LsciPca processor(8, 8, 4, 3);
    EXPECT_THROW(processor.process(), std::logic_error);
    std::vector<float> input(64, 1.f);
    input[0] = -1.f;
    for (int t = 0; t < 4; ++t)
        processor.add_frame(input.data(), camera::PixelDepth::Bits32);
    EXPECT_THROW(processor.process(), std::invalid_argument);
    EXPECT_THROW((holovibes::compute::LsciPca(8, 8, 1, 3)), std::invalid_argument);
    EXPECT_THROW((holovibes::compute::LsciPca(8, 8, 4, 2)), std::invalid_argument);
}

TEST_F(LsciPcaTest, LocalMatchesCpuReconstructionAcrossPatchAndImageBorders)
{
    constexpr unsigned w = 13, h = 11, n = 8, patch = 5, side = 7;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        for (unsigned y = 0; y < h; ++y)
            for (unsigned x = 0; x < w; ++x)
            {
                const unsigned region = (x/patch+y/patch)%3;
                const int checker = (x+y)%2 ? 1 : -1;
                const int spatial_x = static_cast<int>(x%3)-1;
                const int spatial_y = static_cast<int>(y%3)-1;
                const int noise = static_cast<int>((x*13+y*7+t*11)%9)-4;
                input[t*w*h+y*w+x] = static_cast<unsigned char>(120 + checker*(region == 0 ? 35 : 5) +
                    spatial_x*(region == 1 ? 25 : 4)*std::cos(6.283185307179586*t/n) +
                    spatial_y*(region == 2 ? 25 : 3)*std::sin(12.566370614359172*t/n) + noise);
            }
    const auto local = reconstruct_local(input, w, h, n, patch);
    const auto global = reconstruct(input, w*h, n);
    ASSERT_EQ(local.patches.size(), 9u);
    EXPECT_EQ(local.patches.back().width, 3u);
    EXPECT_EQ(local.patches.back().height, 1u);
    double largest_difference = 0.;
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x)
            largest_difference = std::max(largest_difference, std::abs(
                pooled_k(local.images.reconstructed, x, y, w, h, side) -
                pooled_k(global.reconstructed, x, y, w, h, side)));
    // This sequence has different dominant modes in different regions. It must
    // expose an implementation which accidentally reuses one global PCA basis.
    ASSERT_GT(largest_difference, .01);
    compare(input, w, h, n, side, camera::PixelDepth::Bits8, patch);
}

TEST_F(LsciPcaTest, LocalSinglePatchMatchesGlobalMode)
{
    constexpr unsigned w = 9, h = 7, n = 6;
    std::vector<float> input(w*h*n);
    for (unsigned i = 0; i < input.size(); ++i)
        input[i] = 20.f + static_cast<float>((i*17)%71);
    holovibes::compute::LsciPca global(w, h, n, 3);
    holovibes::compute::LsciPca local(w, h, n, 3, 16);
    for (unsigned t = 0; t < n; ++t)
    {
        global.add_frame(input.data()+t*w*h, camera::PixelDepth::Bits32);
        local.add_frame(input.data()+t*w*h, camera::PixelDepth::Bits32);
    }
    const auto original = global.process(), patched = local.process();
    ASSERT_EQ(patched.patches.size(), 1u);
    EXPECT_EQ(patched.patches[0].static_components, original.static_components);
    for (unsigned p = 0; p < w*h; ++p)
    {
        EXPECT_NEAR(patched.original[p], original.original[p], 2e-5);
        EXPECT_NEAR(patched.static_component[p], original.static_component[p], 5e-4);
        EXPECT_NEAR(patched.dynamic[p], original.dynamic[p], 5e-4);
    }
    compare(input, w, h, n, 3, camera::PixelDepth::Bits32, 16);
}

TEST_F(LsciPcaTest, LocalSupports16BitAndFloatIntensities)
{
    constexpr unsigned w = 9, h = 7, n = 6;
    std::vector<unsigned short> shorts(w*h*n);
    std::vector<float> floats(w*h*n);
    for (unsigned i = 0; i < shorts.size(); ++i)
    {
        shorts[i] = static_cast<unsigned short>(1000+(i*977)%50000);
        floats[i] = shorts[i]*.001f;
    }
    compare(shorts, w, h, n, 3, camera::PixelDepth::Bits16, 4);
    compare(floats, w, h, n, 3, camera::PixelDepth::Bits32, 4);
}

TEST_F(LsciPcaTest, LocalStationaryTextureAndZeroSignalHaveNoDynamicContrast)
{
    constexpr unsigned w = 9, h = 7, n = 8;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        for (unsigned p = 0; p < w*h; ++p)
            input[t*w*h+p] = static_cast<unsigned char>(10+(p*17)%180);
    compare(input, w, h, n, 7, camera::PixelDepth::Bits8, 4);
    std::fill(input.begin(), input.end(), 0);
    compare(input, w, h, n, 7, camera::PixelDepth::Bits8, 4);
}

TEST_F(LsciPcaTest, LocalSpatiallyUniformSignalRetainsTemporalMeanWithoutPcaModes)
{
    constexpr unsigned w = 9, h = 7, n = 8;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        std::fill(input.begin()+t*w*h, input.begin()+(t+1)*w*h, static_cast<unsigned char>(21+t*3));
    const auto reference = reconstruct_local(input, w, h, n, 4);
    for (const auto& patch : reference.patches)
        ASSERT_EQ(patch.components, 0u);
    compare(input, w, h, n, 3, camera::PixelDepth::Bits8, 4);
}

TEST_F(LsciPcaTest, RejectsInvalidLocalPatchSizes)
{
    EXPECT_THROW((holovibes::compute::LsciPca(8, 8, 4, 3, 1)), std::invalid_argument);
    EXPECT_THROW((holovibes::compute::LsciPca(8, 8, 4, 3, 513)), std::invalid_argument);
}

TEST_F(LsciPcaTest, LocalLargePatchSpansMultipleProjectionTiles)
{
    constexpr unsigned w = 260, h = 260, n = 4;
    std::vector<unsigned char> input(w*h*n);
    for (unsigned t = 0; t < n; ++t)
        for (unsigned p = 0; p < w*h; ++p)
            input[t*w*h+p] = static_cast<unsigned char>(30+(p*17+t*13)%170);
    // More than 65536 pixels: the final projection tile has a different stride.
    compare(input, w, h, n, 3, camera::PixelDepth::Bits8, 512);
}
}
