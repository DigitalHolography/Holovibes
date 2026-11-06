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
                 camera::PixelDepth depth)
    {
        holovibes::compute::LsciPca processor(w, h, n, side);
        const auto reference = reconstruct(input, w*h, n);
        // Reuse the same processor to check that consecutive blocks do not mix.
        for (unsigned block = 0; block < 2; ++block)
        {
            for (unsigned t = 0; t < n; ++t)
                processor.add_frame(input.data()+t*w*h, depth);
            const auto output = processor.process();
            ASSERT_EQ(output.static_components, reference.components);
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
}
