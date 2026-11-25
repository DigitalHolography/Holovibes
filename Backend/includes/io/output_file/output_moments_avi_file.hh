#pragma once

#include <array>
#include <filesystem>

#include <opencv2/videoio.hpp>

#include "output_frame_file.hh"

namespace holovibes::io_files
{
/*! \brief Exports each plane of an interleaved float moment frame to its own AVI. */
class OutputMomentsAviFile final : public OutputFrameFile
{
  public:
    ~OutputMomentsAviFile() override;

    size_t get_total_nb_frames() const override { return frames_written_; }
    void export_compute_settings(int, size_t) override {}
    void write_header() override;
    size_t write_frame(const char* frame, size_t frame_size) override;
    void write_footer() override;
    void correct_number_of_frames(size_t) override {}

  private:
    friend class OutputFrameFileFactory;
    OutputMomentsAviFile(const std::string& file_path, const camera::FrameDescriptor& fd, bool fft_shift);

    void reserve_output_paths();

    struct Movie
    {
        std::filesystem::path path;
        cv::VideoWriter writer;
        float amplitude = 0.f;
        bool published = false;
    };

    std::array<Movie, 3> movies_;
    std::filesystem::path staging_dir_;
    cv::Mat image_;
    const int shift_x_;
    const int shift_y_;
    size_t frames_written_ = 0;
    bool completed_ = false;
};
} // namespace holovibes::io_files
