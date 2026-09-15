#include "gtest/gtest.h"

#include "async_record_writer.hh"
#include "output_frame_file_factory.hh"
#include "output_holo_file.hh"
#include "test_disable_log.hh"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <system_error>
#include <thread>

namespace
{
constexpr size_t MIB = size_t{1} << 20;

struct WriteGate
{
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool released = false;

    void release()
    {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};

struct FinishRecordingOnExit
{
    WriteGate& gate;
    holovibes::worker::AsyncRecordWriter& writer;
    std::shared_ptr<holovibes::worker::AsyncRecordWriter::Job> job;
    ~FinishRecordingOnExit()
    {
        gate.release();
        writer.finish(job, 1);
    }
};

struct AbortJobOnExit
{
    holovibes::worker::AsyncRecordWriter& writer;
    std::shared_ptr<holovibes::worker::AsyncRecordWriter::Job> job;
    bool active = true;
    ~AbortJobOnExit()
    {
        if (active)
            writer.abort(job, "test exited before capture finished");
    }
};

class SmallOutputFile : public holovibes::io_files::OutputFrameFile
{
  public:
    SmallOutputFile(const std::string& path, WriteGate* gate = nullptr)
        : OutputFrameFile(path)
        , gate_(gate)
    {
    }

    size_t get_total_nb_frames() const override { return frame_count_; }
    void export_compute_settings(int, size_t) override {}
    void write_header() override { std::fputc('H', file_); }
    size_t write_frame(const char* frame, size_t size) override
    {
        if (gate_)
        {
            std::unique_lock lock(gate_->mutex);
            gate_->started = true;
            gate_->changed.notify_all();
            gate_->changed.wait(lock, [&] { return gate_->released; });
        }
        return std::fwrite(frame, 1, size, file_);
    }
    void write_footer() override { std::fputc('F', file_); }
    void correct_number_of_frames(size_t count) override { frame_count_ = count; }

  private:
    WriteGate* gate_;
    size_t frame_count_ = 0;
};

std::unique_ptr<char[]> one_byte(char value)
{
    auto bytes = std::make_unique<char[]>(1);
    bytes[0] = value;
    return bytes;
}

std::string read_all(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
} // namespace

TEST(AsyncRecordWriterTest, BuffersSecondRecordingWhileFirstDiskWriteIsBlocked)
{
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto first_path = std::filesystem::path(::testing::TempDir()) / ("async_first_" + suffix + ".bin");
    const auto second_path = std::filesystem::path(::testing::TempDir()) / ("async_second_" + suffix + ".bin");
    WriteGate gate;
    holovibes::worker::AsyncRecordWriter writer;
    writer.set_limit_gib(1);

    auto first = writer.reserve(700 * MIB);
    ASSERT_NE(first, nullptr);
    FinishRecordingOnExit finish_on_exit{gate, writer, first};
    auto first_file = std::make_unique<SmallOutputFile>(first_path.string(), &gate);
    first_file->write_header();
    writer.attach_file(first, std::move(first_file));
    writer.enqueue(first, one_byte('A'), 1);

    {
        std::unique_lock lock(gate.mutex);
        EXPECT_TRUE(gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.started; }));
    }

    auto second = writer.reserve(300 * MIB);
    ASSERT_NE(second, nullptr);
    auto second_file = std::make_unique<SmallOutputFile>(second_path.string());
    second_file->write_header();
    writer.attach_file(second, std::move(second_file));
    writer.enqueue(second, one_byte('B'), 1);
    writer.finish(second, 1);
    EXPECT_EQ(writer.pending_count(), 2);
    EXPECT_FALSE(writer.can_reserve(25 * MIB));

    gate.release();
    writer.finish(first, 1);
    for (int i = 0; i < 100 && writer.pending_count() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(writer.pending_count(), 0);
    EXPECT_EQ(writer.failed_count(), 0);
    EXPECT_EQ(read_all(first_path), "HAF");
    EXPECT_EQ(read_all(second_path), "HBF");
    EXPECT_TRUE(writer.can_reserve(700 * MIB));

    std::error_code ignored;
    std::filesystem::remove(first_path, ignored);
    std::filesystem::remove(second_path, ignored);
}

TEST(AsyncRecordWriterTest, FailedCaptureReleasesItsReservation)
{
    holovibes::worker::AsyncRecordWriter writer;
    writer.set_limit_gib(1);
    auto job = writer.reserve(1ULL << 30);
    ASSERT_NE(job, nullptr);
    EXPECT_FALSE(writer.can_reserve(1));

    writer.abort(job, "staging allocation failed");
    for (int i = 0; i < 100 && writer.pending_count() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(writer.pending_count(), 0);
    EXPECT_EQ(writer.failed_count(), 1);
    EXPECT_EQ(writer.last_error(), "staging allocation failed");
    EXPECT_TRUE(writer.can_reserve(1ULL << 30));
}

TEST(AsyncRecordWriterTest, ManuallyStoppedCapturesSplitTheBudget)
{
    holovibes::worker::AsyncRecordWriter writer;
    writer.set_limit_gib(1);
    auto first = writer.reserve_unbounded(1);
    auto second = writer.reserve_unbounded(1);
    ASSERT_NE(first, nullptr);
    if (!second)
    {
        writer.abort(first, "test finished");
        FAIL() << "Second manual recording could not reserve half the RAM budget";
        return;
    }
    EXPECT_EQ(first->reserved_bytes, 512 * MIB);
    EXPECT_EQ(second->reserved_bytes, 512 * MIB);
    EXPECT_FALSE(writer.can_reserve_unbounded(1));

    writer.abort(first, "test finished");
    writer.abort(second, "test finished");
}

TEST(AsyncRecordWriterTest, StagingWaitsWhenItsShareIsFull)
{
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::path(::testing::TempDir()) / ("async_wait_" + suffix + ".bin");
    WriteGate gate;
    holovibes::worker::AsyncRecordWriter writer;
    writer.set_limit_gib(1);
    auto job = writer.reserve(1);
    ASSERT_NE(job, nullptr);
    FinishRecordingOnExit finish_on_exit{gate, writer, job};

    auto file = std::make_unique<SmallOutputFile>(path.string(), &gate);
    file->write_header();
    writer.attach_file(job, std::move(file));
    writer.enqueue(job, one_byte('A'), 1);
    {
        std::unique_lock lock(gate.mutex);
        EXPECT_TRUE(gate.changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate.started; }));
    }

    auto second_frame = std::async(std::launch::async, [&] { writer.enqueue(job, one_byte('B'), 1); });
    EXPECT_EQ(second_frame.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    gate.release();
    const auto outcome = second_frame.wait_for(std::chrono::seconds(5));
    EXPECT_EQ(outcome, std::future_status::ready);
    if (outcome != std::future_status::ready)
        writer.abort(job, "staging wait timed out");
    EXPECT_NO_THROW(second_frame.get());
    writer.finish(job, 2);
    for (int i = 0; i < 100 && writer.pending_count() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_EQ(writer.pending_count(), 0);
    EXPECT_EQ(read_all(path), "HABF");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(AsyncRecordWriterTest, HoloFooterIsWrittenAfterCaptureMetadataIsReady)
{
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::path(::testing::TempDir()) / ("async_footer_" + suffix + ".holo");
    constexpr camera::FrameDescriptor fd = {1, 1, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
    holovibes::worker::AsyncRecordWriter writer;
    writer.set_limit_gib(1);
    auto job = writer.reserve(1);
    ASSERT_NE(job, nullptr);
    AbortJobOnExit abort_on_exit{writer, job};

    std::unique_ptr<holovibes::io_files::OutputFrameFile> file(
        holovibes::io_files::OutputFrameFileFactory::create(path.string(), fd, 1, holovibes::RecordedDataType::RAW));
    auto* holo = dynamic_cast<holovibes::io_files::OutputHoloFile*>(file.get());
    ASSERT_NE(holo, nullptr);
    file->write_header();
    writer.attach_file(job, std::move(file));
    writer.enqueue(job, one_byte('A'), 1);

    holo->set_session_timestamps_us(1'000, 2'000, 100, 200, 900, 1'800);
    holo->export_compute_settings(1'000, 1);
    writer.finish(job, 1);
    abort_on_exit.active = false;
    for (int i = 0; i < 100 && writer.pending_count() != 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    ASSERT_EQ(writer.pending_count(), 0);
    std::ifstream saved(path, std::ios::binary);
    saved.seekg(65); // 64-byte holo header and one recorded byte.
    nlohmann::json footer;
    saved >> footer;
    EXPECT_EQ(footer.at("info").at("timestamps_us").at("unix_first"), 1'000);
    EXPECT_EQ(footer.at("info").at("timestamps_us").at("unix_last"), 2'000);

    saved.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}
