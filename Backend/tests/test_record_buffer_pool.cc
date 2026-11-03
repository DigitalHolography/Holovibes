#include "gtest/gtest.h"

#include <chrono>
#include <cuda_runtime.h>
#include <sstream>
#include <spdlog/sinks/ostream_sink.h>
#include <thread>
#include <vector>

#include "API.hh"
#include "fast_updates_holder.hh"
#include "test_disable_log.hh"

namespace holovibes
{
namespace
{
void CUDART_CB delay_recorder_stream(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(1500)); }

class ScopedLogCapture
{
  public:
    explicit ScopedLogCapture(std::ostream& output)
        : logger_(Logger::logger())
        , original_level_(logger_->level())
        , sink_(std::make_shared<spdlog::sinks::ostream_sink_mt>(output))
    {
        sink_->set_pattern("%v");
        logger_->sinks().push_back(sink_);
        logger_->set_level(spdlog::level::info);
    }

    ~ScopedLogCapture()
    {
        logger_->set_level(original_level_);
        std::erase(logger_->sinks(), sink_);
    }

  private:
    std::shared_ptr<spdlog::logger> logger_;
    spdlog::level::level_enum original_level_;
    spdlog::sink_ptr sink_;
};

class RecordBufferPoolTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            GTEST_SKIP() << "Record buffer allocation requires a CUDA device";

        original_count_ = API.record.get_record_buffer_count();
        original_size_ = API.record.get_record_buffer_size();
        original_location_ = API.record.get_record_queue_location();
        original_mode_ = API.record.get_record_mode();
        original_import_ = API.input.get_import_type();
        original_fd_ = API.input.get_input_fd();
        initialized_ = true;

        // Set up a tiny source descriptor without starting the reader or compute workers.
        API.input.set_import_type(ImportType::None);
        API.input.set_input_fd({2, 2, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian});
        API.record.set_record_mode(RecordMode::RAW);
        API.record.set_record_buffer_size(4);
        API.record.set_record_queue_location(Device::CPU);
    }

    void TearDown() override
    {
        if (!initialized_)
            return;

        API.input.set_import_type(ImportType::None);
        API.record.set_record_mode(RecordMode::NONE);
        Holovibes::instance().init_record_queue();

        API.input.set_input_fd(original_fd_);
        API.record.set_record_buffer_count(original_count_);
        API.record.set_record_buffer_size(original_size_);
        API.record.set_record_queue_location(original_location_);
        API.record.set_record_mode(original_mode_);
        API.input.set_import_type(original_import_);
    }

    void expect_pool(uint count, Device location)
    {
        auto& queues = FastUpdatesMap::map<QueueType>;
        for (uint index = 0; index < count; ++index)
        {
            const auto type = record_queue_type(index);
            ASSERT_TRUE(queues.contains(type)) << "Missing record buffer " << index + 1;
            const auto entry = queues.get_entry(type);
            EXPECT_EQ(std::get<0>(*entry).load(), 0u);
            EXPECT_EQ(std::get<1>(*entry).load(), 4u);
            EXPECT_EQ(std::get<2>(*entry).load(), location);
            for (uint previous = 0; previous < index; ++previous)
                EXPECT_NE(entry, queues.get_entry(record_queue_type(previous)));
        }
        for (const auto& [type, entry] : queues)
            if (is_record_queue(type))
                EXPECT_LT(record_queue_index(type), count) << "Stale record buffer telemetry";

        const auto queue = Holovibes::instance().get_record_queue().load();
        ASSERT_NE(queue, nullptr);
        EXPECT_EQ(queue->get_max_size(), 4u);
        EXPECT_TRUE(Holovibes::instance().can_start_frame_record());
        EXPECT_FALSE(API.record.is_recording());
    }

    uint original_count_ = 0;

  private:
    bool initialized_ = false;
    uint original_size_ = 0;
    Device original_location_ = Device::CPU;
    RecordMode original_mode_ = RecordMode::NONE;
    ImportType original_import_ = ImportType::None;
    camera::FrameDescriptor original_fd_{};
};

TEST_F(RecordBufferPoolTest, DefaultsToThree)
{
    EXPECT_EQ(original_count_, 3u);
    Holovibes::instance().init_record_queue();
    expect_pool(3, Device::CPU);
}

TEST_F(RecordBufferPoolTest, GrowsAndShrinksCpuBuffersBeyondThree)
{
    for (const uint count : {2u, 8u, 1u, 5u})
    {
        EXPECT_EQ(API.record.set_record_buffer_count(count), ApiCode::OK);
        Holovibes::instance().init_record_queue();
        expect_pool(count, Device::CPU);
    }

    EXPECT_EQ(API.record.set_record_buffer_count(0), ApiCode::INVALID_VALUE);
    EXPECT_EQ(API.record.get_record_buffer_count(), 5u);
    EXPECT_EQ(API.record.set_record_buffer_count(5), ApiCode::NO_CHANGE);
    expect_pool(5, Device::CPU);
}

TEST_F(RecordBufferPoolTest, AllocatesGpuBuffersAndChangesTheirLocation)
{
    EXPECT_EQ(API.record.set_record_buffer_count(4), ApiCode::OK);
    EXPECT_EQ(API.record.set_record_queue_location(Device::GPU), ApiCode::OK);
    Holovibes::instance().init_record_queue();
    expect_pool(4, Device::GPU);

    EXPECT_EQ(API.record.set_record_queue_location(Device::CPU), ApiCode::OK);
    Holovibes::instance().init_record_queue();
    expect_pool(4, Device::CPU);
}

TEST_F(RecordBufferPoolTest, LogsTheStartAndCompletionOfEachBuffer)
{
    EXPECT_EQ(API.record.set_record_buffer_count(5), ApiCode::OK);
    std::ostringstream output;
    {
        ScopedLogCapture capture(output);
        Holovibes::instance().init_record_queue();
    }

    const auto logs = output.str();
    for (uint index = 1; index <= 5; ++index)
    {
        const auto buffer = std::to_string(index) + "/5";
        EXPECT_NE(logs.find("Allocating record buffer " + buffer), std::string::npos);
        EXPECT_NE(logs.find("Record buffer " + buffer + " allocated"), std::string::npos);
    }
}

TEST_F(RecordBufferPoolTest, ReportsProgressWhileWaitingForRecorderStream)
{
    EXPECT_EQ(API.record.set_record_buffer_count(1), ApiCode::OK);
    EXPECT_EQ(API.record.set_record_queue_location(Device::GPU), ApiCode::OK);
    Holovibes::instance().init_record_queue();

    std::ostringstream output;
    {
        ScopedLogCapture capture(output);
        const auto stream = Holovibes::instance().get_cuda_streams().recorder_stream;
        ASSERT_EQ(cudaLaunchHostFunc(stream, delay_recorder_stream, nullptr), cudaSuccess);
        // Rebuilding waits for this stream, so the allocation reporter must stay active.
        Holovibes::instance().init_record_queue();
    }

    const auto logs = output.str();
    EXPECT_NE(logs.find("Still allocating record buffer 1/1"), std::string::npos);
    EXPECT_NE(logs.find("Record buffer 1/1 allocated"), std::string::npos);
}
} // namespace
} // namespace holovibes
