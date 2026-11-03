#include "gtest/gtest.h"

#include "output_frame_file_factory.hh"
#include "output_holo_file.hh"
#include "test_disable_log.hh"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
constexpr std::streamoff HOLO_HEADER_SIZE = 64;
constexpr camera::FrameDescriptor FRAME_DESCRIPTOR = {
    1, 1, camera::PixelDepth::Bits8, camera::Endianness::LittleEndian};
constexpr std::array<char, 3> FRAME_DATA = {1, 2, 3};

class TemporaryHoloFile
{
  public:
    explicit TemporaryHoloFile(const char* stem)
        : path_(std::filesystem::path(::testing::TempDir()) /
                (std::string(stem) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".holo"))
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    ~TemporaryHoloFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

nlohmann::json write_holo_and_read_footer(const std::filesystem::path& path,
                                          std::vector<holovibes::io_files::FrameTimestampUs> frame_timestamps = {})
{
    std::unique_ptr<holovibes::io_files::OutputFrameFile> output(
        holovibes::io_files::OutputFrameFileFactory::create(path.string(),
                                                            FRAME_DESCRIPTOR,
                                                            FRAME_DATA.size(),
                                                            holovibes::RecordedDataType::RAW));

    auto* holo = dynamic_cast<holovibes::io_files::OutputHoloFile*>(output.get());
    if (holo == nullptr)
        throw std::runtime_error("OutputFrameFileFactory did not create an OutputHoloFile");

    output->write_header();
    for (const char frame : FRAME_DATA)
        output->write_frame(&frame, sizeof(frame));

    if (!frame_timestamps.empty())
    {
        const auto& first = frame_timestamps.front();
        const auto& last = frame_timestamps.back();
        holo->set_session_timestamps_us(first.unix_us,
                                        last.unix_us,
                                        first.camera_us,
                                        last.camera_us,
                                        first.offset_us,
                                        last.offset_us);
        holo->set_frame_timestamps_us(std::move(frame_timestamps));
    }

    output->export_compute_settings(1'000, FRAME_DATA.size());
    output->write_footer();
    output.reset();

    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Unable to reopen temporary holo file");

    file.seekg(HOLO_HEADER_SIZE + static_cast<std::streamoff>(FRAME_DATA.size()));
    nlohmann::json footer;
    file >> footer;
    return footer;
}
} // namespace

TEST(OutputHoloFileTest, PerFrameTimestampsAreAbsentByDefault)
{
    TemporaryHoloFile file("holovibes_output_holo_without_frame_timestamps");

    const auto footer = write_holo_and_read_footer(file.path());
    const auto& timestamps = footer.at("info").at("timestamps_us");

    EXPECT_FALSE(timestamps.contains("per_frame"));
}

TEST(OutputHoloFileTest, SerializesPerFrameTimestampsAndSessionBounds)
{
    TemporaryHoloFile file("holovibes_output_holo_with_frame_timestamps");
    std::vector<holovibes::io_files::FrameTimestampUs> samples = {
        {1'000'000, 100'000, 900'000},
        {1'000'125, 100'125, 900'000},
        {1'000'250, 100'249, 900'001},
    };

    const auto footer = write_holo_and_read_footer(file.path(), samples);
    const auto& timestamps = footer.at("info").at("timestamps_us");

    EXPECT_EQ(timestamps.at("unix_first"), samples.front().unix_us);
    EXPECT_EQ(timestamps.at("unix_last"), samples.back().unix_us);
    EXPECT_EQ(timestamps.at("duration"), samples.back().unix_us - samples.front().unix_us);
    EXPECT_EQ(timestamps.at("camera_first"), samples.front().camera_us);
    EXPECT_EQ(timestamps.at("camera_last"), samples.back().camera_us);
    EXPECT_EQ(timestamps.at("offset_first"), samples.front().offset_us);
    EXPECT_EQ(timestamps.at("offset_last"), samples.back().offset_us);

    const auto& per_frame = timestamps.at("per_frame");
    EXPECT_EQ(per_frame.at("unix"), nlohmann::json::array({1'000'000, 1'000'125, 1'000'250}));
    EXPECT_EQ(per_frame.at("camera"), nlohmann::json::array({100'000, 100'125, 100'249}));
    EXPECT_EQ(per_frame.at("offset"), nlohmann::json::array({900'000, 900'000, 900'001}));
}
