#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <tuple>

#include "test_disable_log.hh"

namespace holovibes
{
namespace
{
TEST(PcaEffectiveRank, LegacySettingsDefaultToOffAndEnabledSettingsRoundTrip)
{
    Rendering rendering;
    rendering.time_transformation = TimeTransformation::PCA;
    const json legacy = rendering;
    EXPECT_FALSE(legacy.contains("pca_effective_rank"));

    rendering.pca_effective_rank = {true, 0.52f};
    const json enabled = rendering;
    EXPECT_EQ(enabled.at("pca_effective_rank").get<Rendering::PcaEffectiveRank>(), rendering.pca_effective_rank);

    from_json(legacy, rendering);
    EXPECT_EQ(rendering.pca_effective_rank, Rendering::PcaEffectiveRank{});

    json invalid = enabled;
    invalid["pca_effective_rank"]["factor"] = -1.0;
    EXPECT_THROW(invalid.get<Rendering>(), std::invalid_argument);
}

TEST(PcaEffectiveRank, SuppressesStrongestClassicPcaScoresAtTheirOriginalIndices)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        GTEST_SKIP() << "Classic PCA requires CUDA";

    constexpr camera::FrameDescriptor fd = {2, 2, camera::PixelDepth::Complex, camera::Endianness::LittleEndian};
    constexpr unsigned frames = 4;
    constexpr std::array amplitudes{1.f, 2.f, 3.f, 8.f};
    std::array<cuComplex, frames * frames> input{};
    for (unsigned t = 0; t < frames; ++t)
        input[t * frames + t].x = amplitudes[t];

    const auto& stream = Holovibes::instance().get_cuda_streams().compute_stream;
    cuda_tools::CublasHandle::set_stream(stream);
    cuda_tools::CusolverHandle::set_stream(stream);

    auto functions = std::make_shared<FunctionVector>();
    CoreBuffersEnv buffers;
    cuda_tools::CufftHandle spatial_plan;
    TimeTransformationEnv time;
    MomentsEnv moments;
    time.gpu_time_transformation_queue = std::make_unique<Queue>(fd, frames);
    time.gpu_p_acc_buffer.resize(input.size());
    time.pca_cov.resize(frames * frames);
    time.pca_eigen_values.resize(frames);
    time.pca_dev_info.resize(1);
    ASSERT_EQ(cudaMemcpyAsync(time.gpu_time_transformation_queue->get_data(),
                              input.data(),
                              sizeof(input),
                              cudaMemcpyHostToDevice,
                              stream),
              cudaSuccess);

    std::tuple<ALL_SETTINGS> initial;
    std::get<settings::TimeTransformation>(initial).value = TimeTransformation::PCA;
    std::get<settings::TimeTransformationSize>(initial).value = frames;
    compute::FourierTransform transform(functions, buffers, fd, spatial_plan, time, moments, stream, initial);
    transform.insert_time_transform();

    const auto compare_scores = [&](Rendering::PcaEffectiveRank options, unsigned removed)
    {
        transform.update_setting(settings::PcaEffectiveRank{options});
        transform.pipe_cycle_apply_updates();
        functions->call_all();

        decltype(input) scores;
        ASSERT_EQ(
            cudaMemcpyAsync(scores.data(), time.gpu_p_acc_buffer.get(), sizeof(scores), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        SCOPED_TRACE(options.factor);
        for (unsigned component = 0; component < frames; ++component)
        {
            for (unsigned pixel = 0; pixel < frames; ++pixel)
            {
                const auto score = scores[component * frames + pixel];
                const float expected = component < frames - removed && component == pixel ? amplitudes[component] : 0.f;
                EXPECT_NEAR(std::hypot(score.x, score.y), expected, 1e-5f);
            }
        }
    };
    compare_scores({}, 0);
    compare_scores({false, 20.f}, 0);
    compare_scores({true, 0.f}, 0);
    // The eigenvalues 1, 4, 9, 64 give an effective rank of about 1.86.
    compare_scores({true, 0.52f}, 1);
    compare_scores({true, 1.f}, 2);
    compare_scores({true, 20.f}, frames);
    compare_scores({}, 0);
}
} // namespace
} // namespace holovibes
