#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>

#include "local_pca.hh"
#include "unique_ptr.hh"
#include "test_disable_log.hh"

namespace
{
using holovibes::compute::LocalPca;
using holovibes::cuda_tools::CudaUniquePtr;

struct ReferenceResult
{
    std::vector<std::complex<double>> samples;
    std::vector<unsigned> coverage;
};

// Independent double-precision CPU oracle. Represent a complex Hermitian Gram
// matrix as the real symmetric matrix [Re(C), -Im(C); Im(C), Re(C)]. Each complex
// eigenvalue appears twice, and the top 2*r real eigenvectors span exactly the
// clutter subspace. This avoids depending on the CUDA eigensolver or its phases.
ReferenceResult reference_local_pca(const std::vector<cuComplex>& input,
                                    unsigned width,
                                    unsigned height,
                                    unsigned frames,
                                    unsigned width_x,
                                    unsigned width_y,
                                    unsigned stride_x,
                                    unsigned stride_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t tile_pixels = static_cast<size_t>(width_x) * width_y;
    ReferenceResult result{std::vector<std::complex<double>>(input.size()), std::vector<unsigned>(pixels)};
    for (unsigned y = 0; y < height; y += stride_y)
        for (unsigned x = 0; x < width; x += stride_x)
        {
            std::vector<std::complex<double>> tile(tile_pixels * frames);
            for (unsigned t = 0; t < frames; ++t)
                for (size_t p = 0; p < tile_pixels; ++p)
                {
                    const size_t source = t * pixels + ((y + p / width_x) % height) * width + (x + p % width_x) % width;
                    tile[t * tile_pixels + p] = {input[source].x, input[source].y};
                }

            cv::Mat gram = cv::Mat::zeros(2 * frames, 2 * frames, CV_64F);
            for (unsigned row = 0; row < frames; ++row)
                for (unsigned column = 0; column < frames; ++column)
                {
                    std::complex<double> value = 0.;
                    for (size_t p = 0; p < tile_pixels; ++p)
                        value += std::conj(tile[row * tile_pixels + p]) * tile[column * tile_pixels + p];
                    gram.at<double>(row, column) = value.real();
                    gram.at<double>(row, column + frames) = -value.imag();
                    gram.at<double>(row + frames, column) = value.imag();
                    gram.at<double>(row + frames, column + frames) = value.real();
                }
            cv::Mat eigenvalues, eigenvectors;
            if (!cv::eigen(gram, eigenvalues, eigenvectors))
                throw std::runtime_error("CPU reference eigensolver failed");
            double total = 0.;
            for (unsigned i = 0; i < frames; ++i)
                total += std::max(0., eigenvalues.at<double>(2 * i));
            unsigned removed = 0;
            if (total > 0.)
            {
                double entropy = 0.;
                for (unsigned i = 0; i < frames; ++i)
                {
                    const double probability = std::max(0., eigenvalues.at<double>(2 * i)) / total;
                    if (probability > 0.)
                        entropy -= probability * std::log(probability);
                }
                removed = static_cast<unsigned>(
                    std::clamp(std::ceil(std::exp(entropy) - 1e-6), 1., static_cast<double>(frames)));
            }

            for (size_t p = 0; p < tile_pixels; ++p)
            {
                const size_t target = ((y + p / width_x) % height) * width + (x + p % width_x) % width;
                ++result.coverage[target];
                std::vector<double> clutter(2 * frames, 0.);
                for (unsigned c = 0; c < 2 * removed; ++c)
                {
                    double score = 0.;
                    // Project the conjugate of this pixel's temporal row.
                    for (unsigned t = 0; t < frames; ++t)
                        score += eigenvectors.at<double>(c, t) * tile[t * tile_pixels + p].real() -
                                 eigenvectors.at<double>(c, t + frames) * tile[t * tile_pixels + p].imag();
                    for (unsigned t = 0; t < 2 * frames; ++t)
                        clutter[t] += score * eigenvectors.at<double>(c, t);
                }
                for (unsigned t = 0; t < frames; ++t)
                    result.samples[t * pixels + target] +=
                        tile[t * tile_pixels + p] - std::complex<double>(clutter[t], -clutter[t + frames]);
            }
        }
    for (size_t i = 0; i < result.samples.size(); ++i)
        result.samples[i] /= result.coverage[i % pixels];
    return result;
}

std::vector<cuComplex> complex_stack(unsigned width, unsigned height, unsigned frames)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    std::vector<cuComplex> input(pixels * frames);
    for (unsigned t = 0; t < frames; ++t)
        for (size_t p = 0; p < pixels; ++p)
        {
            const double phase = .37 * t + .021 * (p % width) * t;
            input[t * pixels + p] =
                make_cuComplex(static_cast<float>(30. * std::cos(phase) + 2. * std::sin(.71 * p + 1.31 * t)),
                               static_cast<float>(30. * std::sin(phase) + 3. * std::cos(.41 * p - .93 * t)));
        }
    return input;
}

class LocalPcaTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            GTEST_SKIP() << "Local PCA numerical tests require CUDA";
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }
    void TearDown() override
    {
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    std::vector<cuComplex> run(const std::vector<cuComplex>& input,
                               unsigned width,
                               unsigned height,
                               unsigned frames,
                               unsigned width_x,
                               unsigned width_y,
                               unsigned stride_x,
                               unsigned stride_y)
    {
        CudaUniquePtr<cuComplex> gpu_input(input.size()), gpu_output(input.size());
        cudaXMemcpyAsync(gpu_input.get(),
                         input.data(),
                         input.size() * sizeof(cuComplex),
                         cudaMemcpyHostToDevice,
                         stream_);
        LocalPca processor(width, height, frames, width_x, width_y, stride_x, stride_y, stream_);
        processor.process(gpu_input.get(), gpu_output.get());
        std::vector<cuComplex> output(input.size());
        cudaXMemcpyAsync(output.data(),
                         gpu_output.get(),
                         output.size() * sizeof(cuComplex),
                         cudaMemcpyDeviceToHost,
                         stream_);
        cudaXStreamSynchronize(stream_);
        return output;
    }

    void compare(unsigned width,
                 unsigned height,
                 unsigned frames,
                 unsigned width_x,
                 unsigned width_y,
                 unsigned stride_x,
                 unsigned stride_y,
                 unsigned expected_coverage = 0)
    {
        const auto input = complex_stack(width, height, frames);
        const auto expected = reference_local_pca(input, width, height, frames, width_x, width_y, stride_x, stride_y);
        const auto output = run(input, width, height, frames, width_x, width_y, stride_x, stride_y);
        double residual_energy = 0.;
        for (size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_NEAR(output[i].x, expected.samples[i].real(), .003) << "sample " << i;
            EXPECT_NEAR(output[i].y, expected.samples[i].imag(), .003) << "sample " << i;
            residual_energy += std::norm(expected.samples[i]);
        }
        // Ensure the oracle exercises retained signal, rather than only all-zero outputs.
        EXPECT_GT(residual_energy, .01);
        for (unsigned count : expected.coverage)
        {
            EXPECT_GT(count, 0u);
            if (expected_coverage)
                EXPECT_EQ(count, expected_coverage);
        }
    }
    cudaStream_t stream_ = nullptr;
};

TEST_F(LocalPcaTest, RejectsInvalidGeometry)
{
    EXPECT_THROW((LocalPca(0, 8, 4, 4, 4, 2, 2, stream_)), std::invalid_argument);
    EXPECT_THROW((LocalPca(8, 8, 0, 4, 4, 2, 2, stream_)), std::invalid_argument);
    EXPECT_THROW((LocalPca(8, 8, 4, 0, 4, 2, 2, stream_)), std::invalid_argument);
    EXPECT_THROW((LocalPca(8, 8, 4, 4, 4, 0, 2, stream_)), std::invalid_argument);
    EXPECT_THROW((LocalPca(8, 8, 4, 4, 4, 5, 2, stream_)), std::invalid_argument);
    EXPECT_THROW((LocalPca(8, 8, 4, 4, 4, 2, 5, stream_)), std::invalid_argument);
}

TEST_F(LocalPcaTest, ZeroSpectrumAndSingleFrameProduceZeroResidual)
{
    for (unsigned frames : {1u, 4u})
    {
        const auto input =
            frames == 1 ? complex_stack(7, 5, 1) : std::vector<cuComplex>(7 * 5 * frames, make_cuComplex(0.f, 0.f));
        for (const auto value : run(input, 7, 5, frames, 4, 3, 2, 2))
        {
            EXPECT_NEAR(value.x, 0., .001);
            EXPECT_NEAR(value.y, 0., .001);
        }
    }
}

TEST_F(LocalPcaTest, EffectiveRankRemovesStrongestModesAndPreservesWeakest)
{
    // Orthogonal spatial/temporal Hadamard patterns give exact eigenvalues
    // proportional to [10000, 9, .01, 0]. Entropy rank lies strictly in (1,2),
    // so ceil selects two clutter modes and must leave the weak third mode.
    const int basis[3][4] = {{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}};
    std::vector<cuComplex> input(16);
    for (unsigned t = 0; t < 4; ++t)
        for (unsigned p = 0; p < 4; ++p)
        {
            const float value = 100.f + 3.f * basis[1][p] * basis[1][t] + .1f * basis[2][p] * basis[2][t];
            // Distinct complex spatial phases also exercise conjugate projection.
            input[t * 4 + p] = p % 2 ? make_cuComplex(0.f, value) : make_cuComplex(value, 0.f);
        }
    const auto output = run(input, 2, 2, 4, 2, 2, 2, 2);
    for (unsigned t = 0; t < 4; ++t)
        for (unsigned p = 0; p < 4; ++p)
        {
            const double expected = .1 * basis[2][p] * basis[2][t];
            EXPECT_NEAR(output[t * 4 + p].x, p % 2 ? 0. : expected, .003);
            EXPECT_NEAR(output[t * 4 + p].y, p % 2 ? expected : 0., .003);
        }
}

TEST_F(LocalPcaTest, WrapsBothBordersWithUnevenRectangularCoverage) { compare(7, 5, 6, 4, 3, 3, 2); }

TEST_F(LocalPcaTest, AveragesFourOverlappingTilesFor32By32With16By16Strides) { compare(64, 64, 6, 32, 32, 16, 16, 4); }

TEST_F(LocalPcaTest, NonoverlappingTilesHaveUnitCoverage) { compare(8, 6, 6, 4, 3, 4, 3, 1); }

TEST_F(LocalPcaTest, OversizedTilesWrapAndCountRepeatedSamples) { compare(3, 2, 6, 5, 4, 2, 3); }
} // namespace
