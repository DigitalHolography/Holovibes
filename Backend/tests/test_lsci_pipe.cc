#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iterator>
#include <string>
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

PipeSettings lsci_settings(unsigned batch_size)
{
    PipeSettings initial;
    std::apply([](auto&... setting)
               { ((setting = Holovibes::instance().get_setting<std::remove_reference_t<decltype(setting)>>()), ...); },
               initial);

    std::get<settings::ComputeMode>(initial).value = Computation::Hologram;
    std::get<settings::TimeTransformation>(initial).value = TimeTransformation::LSCI;
    std::get<settings::SpaceTransformation>(initial).value = SpaceTransformation::NONE;
    std::get<settings::TimeTransformationSize>(initial).value = 16;
    std::get<settings::LsciSpatialWindow>(initial).value = 5;
    // Deliberately larger than the smaller batches: LSCI must retain every frame,
    // independently of the existing Fourier pipeline's input-decimation gate.
    std::get<settings::TimeStride>(initial).value = 16;
    std::get<settings::BatchSize>(initial).value = batch_size;
    std::get<settings::ImageType>(initial).value = ImgType::Modulus;
    std::get<settings::OutputBufferSize>(initial).value = 8;
    std::get<settings::InputBufferSize>(initial).value = 64;
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
    ViewWindow filter_view;
    filter_view.log_enabled = false;
    filter_view.contrast.enabled = false;
    filter_view.contrast.auto_refresh = false;
    std::get<settings::Filter2d>(initial).value = filter_view;
    return initial;
}

class InspectableLsciPipe : public Pipe
{
  public:
    using Pipe::Pipe;

    bool uses_scalar_storage() const
    {
        return time_transformation_env_.gpu_lsci_moments.get() != nullptr &&
               time_transformation_env_.gpu_time_transformation_queue == nullptr &&
               time_transformation_env_.gpu_p_acc_buffer.get() == nullptr &&
               time_transformation_env_.gpu_oct_float_buffer.get() == nullptr &&
               moments_env_.stft_res_buffer.get() == nullptr;
    }

    bool has_spatial_buffer() const { return buffers_.gpu_spatial_transformation_buffer.get() != nullptr; }
    unsigned partial_frame_count() const { return time_transformation_env_.lsci_frame_count; }
};

// Keep worker cleanup independent of Google Test's early ASSERT returns. In
// particular, a partial producer batch must release its mutex before destruction.
class PipeRunner
{
  public:
    PipeRunner(Pipe& pipe, BatchInputQueue& input)
        : pipe_(pipe)
        , input_(input)
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
                finished_.store(true, std::memory_order_release);
            });
    }

    ~PipeRunner() { stop(); }

    void stop()
    {
        if (worker_.joinable())
        {
            pipe_.request(ICS::Termination);
            worker_.join();
            input_.stop_producer();
        }
    }

    bool finished() const { return finished_.load(std::memory_order_acquire); }

    void rethrow_failure() const
    {
        if (failure_)
            std::rethrow_exception(failure_);
    }

  private:
    Pipe& pipe_;
    BatchInputQueue& input_;
    std::thread worker_;
    std::atomic<bool> finished_{false};
    std::exception_ptr failure_;
};

template <typename Predicate>
bool wait_for_pipe(const PipeRunner& runner, Predicate condition)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!condition())
    {
        if (runner.finished() || std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::yield();
    }
    return true;
}

// Every frame is spatially uniform, making the known population contrast depend
// solely on all 16 temporal samples. Spatial pooling must preserve this value.
void enqueue_uniform_frames(BatchInputQueue& input, const std::vector<unsigned char>& intensities)
{
    if (intensities.empty())
        return;
    const size_t resolution = input.get_fd().get_frame_res();
    std::vector<unsigned char> frames(resolution * intensities.size());
    for (size_t frame = 0; frame < intensities.size(); ++frame)
        std::fill_n(frames.begin() + frame * resolution, resolution, intensities[frame]);
    for (size_t frame = 0; frame < intensities.size(); ++frame)
        input.enqueue(frames.data() + frame * resolution, cudaMemcpyHostToDevice);
    input.sync_pending_copies();
}

double population_contrast(const std::vector<unsigned char>& intensities)
{
    double mean = 0.;
    for (unsigned char value : intensities)
        mean += value;
    mean /= intensities.size();
    double variance = 0.;
    for (unsigned char value : intensities)
        variance += (value - mean) * (value - mean);
    variance /= intensities.size();
    return std::sqrt(variance) / mean;
}

class LsciPipeTest : public testing::TestWithParam<unsigned>
{
  protected:
    void SetUp() override
    {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "LSCI pipeline tests require a CUDA device";
    }
};

TEST_P(LsciPipeTest, ProducesOnlyCompleteConsecutiveNonoverlappingBlocks)
{
    const unsigned batch = GetParam();
    constexpr camera::FrameDescriptor fd = {7, 5, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> active_recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLsciPipe pipe(input, active_recording, stream, lsci_settings(batch));
    auto output = pipe.get_output_queue();
    PipeRunner runner(pipe, input);

    std::vector<unsigned char> first(16), second(16);
    for (unsigned index = 0; index < 16; ++index)
    {
        first[index] = static_cast<unsigned char>(index + 1);
        second[index] = static_cast<unsigned char>(index % 2 == 0 ? 80 : 120);
    }

    const unsigned prefix = 16 - batch;
    enqueue_uniform_frames(input, {first.begin(), first.begin() + prefix});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    EXPECT_EQ(output->get_size(), 0u) << "A temporal prefix must not produce a map";
    enqueue_uniform_frames(input, {first.begin() + prefix, first.end()});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() >= 1; }));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    EXPECT_EQ(output->get_size(), 1u);

    enqueue_uniform_frames(input, {second.begin(), second.begin() + prefix});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    EXPECT_EQ(output->get_size(), 1u) << "Successive blocks must not overlap";
    enqueue_uniform_frames(input, {second.begin() + prefix, second.end()});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() >= 2; }));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));

    // For batch 16 this also tests stopping with an incomplete producer batch;
    // the smaller batches test a consumed but incomplete temporal block.
    const unsigned tail_frames = batch == 16 ? 8 : batch;
    enqueue_uniform_frames(input, std::vector<unsigned char>(tail_frames, 7));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
    ASSERT_EQ(output->get_size(), 2u) << "A partial tail must not produce or reuse a map";
    EXPECT_TRUE(pipe.uses_scalar_storage());

    for (const auto& block : {first, second})
    {
        std::vector<unsigned short> pixels(fd.get_frame_res());
        ASSERT_EQ(output->dequeue(pixels.data(), stream, cudaMemcpyDeviceToHost), 1);
        const double expected = std::floor(65535. * population_contrast(block));
        for (unsigned short value : pixels)
            EXPECT_NEAR(value, expected, 1.) << "The map must use exactly this block's 16 samples";
    }
}

TEST_P(LsciPipeTest, AngularSpectrumPoolsReconstructedIntensityWithoutAllocatingTemporalCubes)
{
    const unsigned batch = GetParam();
    constexpr camera::FrameDescriptor fd = {8, 8, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> active_recording{nullptr};
    const cudaStream_t stream = 0;
    auto initial = lsci_settings(batch);
    std::get<settings::SpaceTransformation>(initial).value = SpaceTransformation::ANGULARSP;
    // Zero-distance angular propagation is the identity on the complex field.
    // LSCI must therefore pool squared input amplitudes, unlike the direct-intensity path.
    std::get<settings::ZDistance>(initial).value = 0.f;
    InspectableLsciPipe pipe(input, active_recording, stream, initial);
    auto output = pipe.get_output_queue();
    PipeRunner runner(pipe, input);

    std::vector<unsigned char> amplitudes(16);
    for (unsigned index = 0; index < amplitudes.size(); ++index)
        amplitudes[index] = static_cast<unsigned char>(index % 2 == 0 ? 80 : 120);
    const unsigned prefix = 16 - batch;
    enqueue_uniform_frames(input, {amplitudes.begin(), amplitudes.begin() + prefix});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    EXPECT_EQ(output->get_size(), 0u);
    enqueue_uniform_frames(input, {amplitudes.begin() + prefix, amplitudes.end()});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() == 1; }));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
    EXPECT_TRUE(pipe.uses_scalar_storage());
    EXPECT_TRUE(pipe.has_spatial_buffer());

    std::vector<unsigned short> pixels(fd.get_frame_res());
    ASSERT_EQ(output->dequeue(pixels.data(), stream, cudaMemcpyDeviceToHost), 1);
    const double low = 80. * 80.;
    const double high = 120. * 120.;
    const double expected = std::floor(65535. * (high - low) / (high + low));
    for (unsigned short value : pixels)
        EXPECT_NEAR(value, expected, 2.);
}

TEST(LsciPipeRegression, QueueResetDiscardsPartialBlockWhileWaitingForInput)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "LSCI pipeline tests require a CUDA device";

    constexpr unsigned batch = 4;
    constexpr camera::FrameDescriptor fd = {7, 5, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> active_recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLsciPipe pipe(input, active_recording, stream, lsci_settings(batch));
    auto output = pipe.get_output_queue();
    PipeRunner runner(pipe, input);

    // The first eight frames have already left the input queue, but remain in
    // the incomplete temporal ensemble while the pipe waits for more input.
    enqueue_uniform_frames(input, std::vector<unsigned char>(8, 5));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    ASSERT_EQ(output->get_size(), 0u);
    input.request_reset();
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return !input.has_pending_reset(); }));

    std::vector<unsigned char> fresh(16);
    for (unsigned index = 0; index < fresh.size(); ++index)
        fresh[index] = static_cast<unsigned char>(index % 2 == 0 ? 80 : 120);
    enqueue_uniform_frames(input, {fresh.begin(), fresh.begin() + 12});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    EXPECT_EQ(output->get_size(), 0u) << "Pre-reset samples must not complete the fresh block";
    enqueue_uniform_frames(input, {fresh.begin() + 12, fresh.end()});
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() >= 1; }));
    ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
    runner.stop();
    ASSERT_NO_THROW(runner.rethrow_failure());
    ASSERT_EQ(output->get_size(), 1u);

    std::vector<unsigned short> pixels(fd.get_frame_res());
    ASSERT_EQ(output->dequeue(pixels.data(), stream, cudaMemcpyDeviceToHost), 1);
    const double expected = std::floor(65535. * population_contrast(fresh));
    for (unsigned short value : pixels)
        EXPECT_NEAR(value, expected, 1.) << "The map must contain only the 16 post-reset frames";
}

TEST(LsciPipeRegression, ReconstructionAndOpticsChangesDiscardPartialBlocksAndRefreshSpatialStorage)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "LSCI pipeline tests require a CUDA device";

    constexpr unsigned batch = 4;
    constexpr camera::FrameDescriptor fd = {8, 8, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> active_recording{nullptr};
    const cudaStream_t stream = 0;
    auto initial = lsci_settings(batch);
    std::get<settings::ZDistance>(initial).value = 0.01f;
    std::get<settings::Lambda>(initial).value = 852e-9f;
    std::get<settings::PixelSize>(initial).value = 12.f;
    InspectableLsciPipe pipe(input, active_recording, stream, initial);
    auto output = pipe.get_output_queue();
    {
        PipeRunner runner(pipe, input);
        enqueue_uniform_frames(input, std::vector<unsigned char>(8, 5));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        runner.stop();
        ASSERT_NO_THROW(runner.rethrow_failure());
    }
    ASSERT_EQ(pipe.partial_frame_count(), 8u);
    EXPECT_FALSE(pipe.has_spatial_buffer());

    const char* changes[] = {"Fresnel",
                             "Angular spectrum",
                             "None",
                             "Angular spectrum again",
                             "Wavelength",
                             "Propagation distance",
                             "Pixel size"};
    for (unsigned stage = 0; stage < std::size(changes); ++stage)
    {
        SCOPED_TRACE(changes[stage]);
        switch (stage)
        {
        case 0:
            pipe.update_setting(settings::SpaceTransformation{SpaceTransformation::FRESNELTR});
            break;
        case 1:
        case 3:
            pipe.update_setting(settings::SpaceTransformation{SpaceTransformation::ANGULARSP});
            break;
        case 2:
            pipe.update_setting(settings::SpaceTransformation{SpaceTransformation::NONE});
            break;
        case 4:
            pipe.update_setting(settings::Lambda{800e-9f});
            break;
        case 5:
            pipe.update_setting(settings::ZDistance{0.02f});
            break;
        case 6:
            pipe.update_setting(settings::PixelSize{13.f});
            break;
        }
        pipe.request(ICS::Refresh);
        pipe.clear_request(ICS::Termination);
        PipeRunner runner(pipe, input);
        enqueue_uniform_frames(input, std::vector<unsigned char>(12, 80));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        EXPECT_EQ(output->get_size(), 0u) << "Samples from the old reconstruction must be discarded";
        enqueue_uniform_frames(input, std::vector<unsigned char>(4, 120));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() == 1; }));
        enqueue_uniform_frames(input, std::vector<unsigned char>(8, 5));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        runner.stop();
        ASSERT_NO_THROW(runner.rethrow_failure());
        EXPECT_EQ(pipe.partial_frame_count(), 8u);
        EXPECT_EQ(pipe.has_spatial_buffer(), stage != 2);
        EXPECT_TRUE(pipe.uses_scalar_storage());
        std::vector<unsigned short> pixels(fd.get_frame_res());
        ASSERT_EQ(output->dequeue(pixels.data(), stream, cudaMemcpyDeviceToHost), 1);
        EXPECT_EQ(output->get_size(), 0u);
    }
}

TEST(LsciPipeRegression, ChangingSpatialWindowUsesNewNeighborhoodAndDiscardsPartialBlock)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "LSCI pipeline tests require a CUDA device";

    constexpr unsigned batch = 4;
    constexpr camera::FrameDescriptor fd = {8, 8, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    BatchInputQueue input(64, batch, fd);
    std::atomic<RecordingContext*> active_recording{nullptr};
    const cudaStream_t stream = 0;
    InspectableLsciPipe pipe(input, active_recording, stream, lsci_settings(batch));
    auto output = pipe.get_output_queue();
    {
        PipeRunner runner(pipe, input);
        enqueue_uniform_frames(input, std::vector<unsigned char>(8, 5));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        runner.stop();
        ASSERT_NO_THROW(runner.rethrow_failure());
    }
    ASSERT_EQ(pipe.partial_frame_count(), 8u);

    std::vector<unsigned char> frame(fd.get_frame_res());
    for (unsigned y = 0; y < fd.height; ++y)
        for (unsigned x = 0; x < fd.width; ++x)
            frame[y * fd.width + x] = (x + y) % 2 == 0 ? 40 : 100;
    auto enqueue_repeated = [&](unsigned frames)
    {
        for (unsigned t = 0; t < frames; ++t)
            input.enqueue(frame.data(), cudaMemcpyHostToDevice);
        input.sync_pending_copies();
    };

    for (unsigned spatial_window : {1u, 3u})
    {
        // Preserve the same Pipe and output queue, changing the setting at a
        // deterministic batch boundary before exercising its normal refresh.
        pipe.update_setting(settings::LsciSpatialWindow{spatial_window});
        pipe.request(ICS::UpdateLsciSpatialWindow);
        pipe.clear_request(ICS::Termination);
        PipeRunner runner(pipe, input);
        enqueue_repeated(12);
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        EXPECT_EQ(output->get_size(), 0u) << "Old-window samples must not complete the new block";
        enqueue_repeated(4);
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return output->get_size() == 1; }));
        enqueue_uniform_frames(input, std::vector<unsigned char>(8, 5));
        ASSERT_TRUE(wait_for_pipe(runner, [&]() { return input.is_empty(); }));
        runner.stop();
        ASSERT_NO_THROW(runner.rethrow_failure());
        EXPECT_EQ(pipe.partial_frame_count(), 8u);
        EXPECT_FALSE(pipe.has_spatial_buffer());

        std::vector<unsigned short> pixels(fd.get_frame_res());
        ASSERT_EQ(output->dequeue(pixels.data(), stream, cudaMemcpyDeviceToHost), 1);
        const int radius = static_cast<int>(spatial_window / 2);
        for (int y = 0; y < fd.height; ++y)
            for (int x = 0; x < fd.width; ++x)
            {
                // The input is constant in time, so independently pool the
                // original spatial samples and calculate centered variance.
                std::vector<double> samples;
                for (int row = std::max(0, y - radius); row <= std::min(int(fd.height) - 1, y + radius); ++row)
                    for (int column = std::max(0, x - radius); column <= std::min(int(fd.width) - 1, x + radius);
                         ++column)
                        samples.push_back(frame[row * fd.width + column]);
                double mean = 0.;
                for (double value : samples)
                    mean += value;
                mean /= samples.size();
                double variance = 0.;
                for (double value : samples)
                    variance += (value - mean) * (value - mean);
                variance /= samples.size();
                const double expected = std::floor(65535. * std::sqrt(variance) / mean);
                EXPECT_NEAR(pixels[y * fd.width + x], expected, 2.);
            }
    }
}

INSTANTIATE_TEST_SUITE_P(ConsecutiveFrames,
                         LsciPipeTest,
                         testing::Values(1u, 4u, 16u),
                         [](const testing::TestParamInfo<unsigned>& info)
                         { return "Batch" + std::to_string(info.param); });
} // namespace
