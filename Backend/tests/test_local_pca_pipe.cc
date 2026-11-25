#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include "holovibes.hh"
#include "test_disable_log.hh"

namespace
{
using namespace holovibes;
using PipeSettings = std::tuple<ALL_SETTINGS>;

PipeSettings local_pca_settings(unsigned batch)
{
    PipeSettings initial;
    std::apply([](auto&... setting)
               { ((setting = Holovibes::instance().get_setting<std::remove_reference_t<decltype(setting)>>()), ...); },
               initial);
    std::get<settings::ComputeMode>(initial).value = Computation::Hologram;
    std::get<settings::TimeTransformation>(initial).value = TimeTransformation::LOCAL_PCA;
    std::get<settings::SpaceTransformation>(initial).value = SpaceTransformation::NONE;
    std::get<settings::TimeTransformationSize>(initial).value = 8;
    std::get<settings::TimeStride>(initial).value = 8;
    std::get<settings::BatchSize>(initial).value = batch;
    std::get<settings::PcaWidthX>(initial).value = 4;
    std::get<settings::PcaWidthY>(initial).value = 3;
    std::get<settings::PcaStrideX>(initial).value = 2;
    std::get<settings::PcaStrideY>(initial).value = 2;
    std::get<settings::PcaRankFactor>(initial).value = 1.f;
    std::get<settings::P>(initial).value = ViewPQ{};
    std::get<settings::ImageType>(initial).value = ImgType::Modulus;
    std::get<settings::OutputBufferSize>(initial).value = 8;
    std::get<settings::DataType>(initial).value = RecordedDataType::RAW;
    std::get<settings::FrameAcquisitionEnabled>(initial).value = false;
    std::get<settings::RecordMode>(initial).value = RecordMode::RAW;
    std::get<settings::ChartDisplayEnabled>(initial).value = false;
    std::get<settings::ChartRecordEnabled>(initial).value = false;
    std::get<settings::LensViewEnabled>(initial).value = false;
    std::get<settings::RawViewEnabled>(initial).value = false;
    std::get<settings::CutsViewEnabled>(initial).value = false;
    std::get<settings::Filter2dEnabled>(initial).value = false;
    std::get<settings::Filter2dViewEnabled>(initial).value = false;
    std::get<settings::FftShiftEnabled>(initial).value = false;
    std::get<settings::RegistrationEnabled>(initial).value = false;
    std::get<settings::RenormEnabled>(initial).value = false;
    std::get<settings::ConvolutionMatrix>(initial).value.clear();
    std::get<settings::InputFilter>(initial).value.clear();
    ViewXYZ view;
    view.log_enabled = false;
    view.contrast.enabled = false;
    view.contrast.auto_refresh = false;
    view.output_image_accumulation = 1;
    std::get<settings::XY>(initial).value = view;
    std::get<settings::XZ>(initial).value = view;
    std::get<settings::YZ>(initial).value = view;
    return initial;
}

class InspectableLocalPcaPipe : public Pipe
{
  public:
    using Pipe::Pipe;
    unsigned partial_frames() const { return time_transformation_env_.gpu_time_transformation_queue->get_size(); }
};

// Always join the compute worker, including after a failed ASSERT.
class LocalPcaRunner
{
  public:
    explicit LocalPcaRunner(Pipe& pipe)
        : pipe_(pipe)
    {
        pipe_.request(ICS::Start);
        worker_ = std::thread(
            [this]()
            {
                try
                {
                    pipe_.exec();
                }
                catch (...)
                {
                    failure_ = std::current_exception();
                }
                finished_.store(true);
            });
    }
    ~LocalPcaRunner() { stop(); }
    void stop()
    {
        if (worker_.joinable())
        {
            pipe_.request(ICS::Termination);
            worker_.join();
        }
    }
    template <typename Predicate>
    bool wait(Predicate predicate) const
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!predicate())
        {
            if (finished_.load() || std::chrono::steady_clock::now() >= deadline)
                return false;
            std::this_thread::yield();
        }
        return true;
    }
    void rethrow_failure() const
    {
        if (failure_)
            std::rethrow_exception(failure_);
    }

  private:
    Pipe& pipe_;
    std::thread worker_;
    std::atomic<bool> finished_{false};
    std::exception_ptr failure_;
};

void enqueue_frames(BatchInputQueue& input, unsigned count, unsigned offset = 0)
{
    const size_t pixels = input.get_fd().get_frame_res();
    std::vector<unsigned char> frames(pixels * count);
    for (unsigned t = 0; t < count; ++t)
        for (size_t p = 0; p < pixels; ++p)
            frames[t * pixels + p] = static_cast<unsigned char>(20 + (p * 7 + (t + offset) * 11) % 180);
    input.enqueue(frames.data(), cudaMemcpyHostToDevice, count);
    input.sync_pending_copies();
}

class LocalPcaPipeTest : public testing::TestWithParam<unsigned>
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            GTEST_SKIP() << "Local PCA pipeline tests require CUDA";
    }
};

TEST_P(LocalPcaPipeTest, UsesEveryFrameAndEmitsOnlyDisjointCompleteBlocks)
{
    const unsigned batch = GetParam();
    constexpr camera::FrameDescriptor fd = {7, 5, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLocalPcaPipe pipe(input, recording, stream, local_pca_settings(batch));
    const auto output = pipe.get_output_queue();
    LocalPcaRunner runner(pipe);

    for (unsigned block = 0; block < 2; ++block)
    {
        for (unsigned consumed = 0; consumed < 8; consumed += batch)
        {
            enqueue_frames(input, batch, block * 8 + consumed);
            const unsigned total = consumed + batch;
            if (total < 8)
            {
                ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == total; }));
                EXPECT_EQ(output->get_size(), block) << "A partial block must not produce or reuse a frame";
            }
            else
                ASSERT_TRUE(runner.wait([&]() { return output->get_size() == block + 1; }));
        }
        ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == 0; }));
    }
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
    EXPECT_EQ(output->get_size(), 2u);
}

TEST_P(LocalPcaPipeTest, GeometryChangeDiscardsPartialTemporalBlock)
{
    const unsigned batch = GetParam();
    if (batch == 8)
        GTEST_SKIP() << "A batch equal to the window has no partial temporal block";
    constexpr camera::FrameDescriptor fd = {7, 5, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLocalPcaPipe pipe(input, recording, stream, local_pca_settings(batch));
    const auto output = pipe.get_output_queue();
    LocalPcaRunner runner(pipe);
    enqueue_frames(input, batch);
    ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == batch; }));
    pipe.update_setting(settings::PcaWidthX{3});
    pipe.request(ICS::UpdateLocalPca);
    // Pipe requests are applied between input batches; release the input wait.
    enqueue_frames(input, batch, 100);
    ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == 0; }));
    enqueue_frames(input, 8 - batch, batch);
    ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == 8 - batch; }));
    EXPECT_EQ(output->get_size(), 0u);
    enqueue_frames(input, batch, 8);
    ASSERT_TRUE(runner.wait([&]() { return output->get_size() == 1; }));
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
}

TEST_P(LocalPcaPipeTest, RankFactorChangesReachLiveComputeWithoutRebuildingPipe)
{
    const unsigned batch = GetParam();
    constexpr camera::FrameDescriptor fd = {7, 5, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    auto initial = local_pca_settings(batch);
    std::get<settings::PcaRankFactor>(initial).value = 0.f;
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLocalPcaPipe pipe(input, recording, stream, initial);
    const auto output = pipe.get_output_queue();
    LocalPcaRunner runner(pipe);

    enqueue_frames(input, 8);
    ASSERT_TRUE(runner.wait([&]() { return output->get_size() == 1; }));
    ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == 0; }));
    std::vector<unsigned short> original(fd.get_frame_res());
    ASSERT_EQ(output->dequeue(original.data(), stream, cudaMemcpyDeviceToHost), 1);
    ASSERT_TRUE(std::any_of(original.begin(), original.end(), [](unsigned short value) { return value > 0; }));

    pipe.update_setting(settings::PcaRankFactor{10.f});
    // A blocked input wait must finish a cycle before cycle settings are applied.
    // Allow one transition block, including the batch == window case.
    enqueue_frames(input, 8, 8);
    ASSERT_TRUE(runner.wait([&]() { return output->get_size() == 1; }));
    ASSERT_TRUE(runner.wait([&]() { return pipe.partial_frames() == 0; }));
    output->clear();
    enqueue_frames(input, 8, 16);
    ASSERT_TRUE(runner.wait([&]() { return output->get_size() == 1; }));
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
    std::vector<unsigned short> removed(fd.get_frame_res());
    ASSERT_EQ(output->dequeue(removed.data(), stream, cudaMemcpyDeviceToHost), 1);
    for (unsigned short value : removed)
        EXPECT_EQ(value, 0u) << "Factor 10 must remove all eight temporal components";
}

TEST(GlobalPcaPipeTest, EffectiveRankKeepsPInChronologicalOrder)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        GTEST_SKIP() << "Global PCA pipeline tests require CUDA";

    constexpr camera::FrameDescriptor fd = {8, 4, camera::PixelDepth::Bits16, camera::Endianness::LittleEndian};
    constexpr unsigned window = 4;
    constexpr std::array<int, window> weak_mode = {200, -400, 800, -600};
    constexpr unsigned input_count = 2 * window;
    constexpr unsigned output_count = input_count - window + 1;
    constexpr unsigned p_start = 1;
    constexpr int background = 12000;
    const size_t pixels = fd.get_frame_res();

    // Each window contains orthogonal static and alternating spatial modes.
    std::vector<unsigned short> frames(input_count * pixels);
    for (unsigned t = 0; t < input_count; ++t)
    {
        const int weak = weak_mode[t % window];
        for (size_t pixel = 0; pixel < pixels; ++pixel)
            frames[t * pixels + pixel] = static_cast<unsigned short>(background + (pixel % 2 == 0 ? weak : -weak));
    }

    for (const float factor : {0.f, 0.5f})
    {
        SCOPED_TRACE(factor);
        auto initial = local_pca_settings(1);
        std::get<settings::TimeTransformation>(initial).value = TimeTransformation::PCA;
        std::get<settings::TimeTransformationSize>(initial).value = window;
        std::get<settings::TimeStride>(initial).value = 1;
        std::get<settings::PcaEffectiveRankEnabled>(initial).value = true;
        std::get<settings::PcaEffectiveRankFactor>(initial).value = factor;
        std::get<settings::P>(initial).value.start = p_start;
        BatchInputQueue input(64, 1, fd);
        std::atomic<RecordingContext*> recording{nullptr};
        const cudaStream_t stream = 0;
        Pipe pipe(input, recording, stream, initial);
        const auto output = pipe.get_output_queue();
        LocalPcaRunner runner(pipe);
        for (unsigned t = 0; t < input_count; ++t)
            input.enqueue(frames.data() + t * pixels, cudaMemcpyHostToDevice);
        input.sync_pending_copies();
        ASSERT_TRUE(runner.wait([&]() { return output->get_size() == output_count; }));
        runner.stop();
        ASSERT_NO_THROW(runner.rethrow_failure());

        std::vector<unsigned short> result(pixels);
        for (unsigned t = 0; t < output_count; ++t)
        {
            SCOPED_TRACE(t);
            ASSERT_EQ(output->dequeue(result.data(), stream, cudaMemcpyDeviceToHost), 1);
            const int weak = weak_mode[(t + p_start) % window];
            EXPECT_NEAR(result.front(), factor == 0.f ? background + weak : std::abs(weak), 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ConsecutiveFrames, LocalPcaPipeTest, testing::Values(1u, 2u, 8u));
} // namespace
