#include "output_moments_avi_file.hh"

#include <algorithm>
#include <cmath>

#include "file_exception.hh"

namespace holovibes::io_files
{
namespace fs = std::filesystem;

OutputMomentsAviFile::OutputMomentsAviFile(const std::string& file_path,
                                           const camera::FrameDescriptor& fd,
                                           bool fft_shift)
    : OutputFrameFile(file_path, OpeningMode::DEFERRED)
    , shift_x_(fft_shift ? fd.width / 2 : 0)
    , shift_y_(fft_shift ? fd.height / 2 : 0)
{
    fd_ = fd;
}

OutputMomentsAviFile::~OutputMomentsAviFile()
{
    if (staging_dir_.empty())
        return;

    std::error_code error;
    for (auto& movie : movies_)
    {
        movie.writer.release();
        fs::remove(staging_dir_ / movie.path.filename(), error);
        if (movie.published && !completed_)
            fs::remove(movie.path, error);
    }
    fs::remove(staging_dir_, error);
}

void OutputMomentsAviFile::reserve_output_paths()
{
    const fs::path requested(file_path_);
    for (size_t suffix = 0;; ++suffix)
    {
        std::string stem = requested.stem().string();
        if (suffix)
            stem += "_" + std::to_string(suffix);

        for (size_t moment = 0; moment < movies_.size(); ++moment)
            movies_[moment].path = requested.parent_path() / (stem + "_m" + std::to_string(moment) + ".avi");

        // An exclusive staging directory reserves the triplet name for concurrent exports.
        const auto staging_dir = requested.parent_path() / (stem + ".moments.tmp");
        if (!fs::create_directory(staging_dir))
            continue;
        staging_dir_ = staging_dir;
        if (std::none_of(movies_.begin(), movies_.end(), [](const auto& movie) { return fs::exists(movie.path); }))
            return;

        fs::remove(staging_dir_);
        staging_dir_.clear();
    }
}

void OutputMomentsAviFile::write_header()
{
    if (fd_.depth != camera::PixelDepth::Bits32 || fd_.width == 0 || fd_.height == 0)
        throw FileException("Moment AVI export requires nonempty float images", false);

    reserve_output_paths();
    try
    {
        // Pad each odd dimension by one pixel to keep the entire image in MJPEG.
        image_ = cv::Mat::zeros(fd_.height + fd_.height % 2, fd_.width + fd_.width % 2, CV_8UC1);
        const double output_fps = compute_output_fps();
        const double fps = output_fps > 0.0 ? output_fps : 24.0;
        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        for (auto& movie : movies_)
        {
            const auto path = staging_dir_ / movie.path.filename();
            if (!movie.writer.open(path.string(), fourcc, fps, image_.size(), false))
                throw FileException("Unable to create moment AVI " + movie.path.string(), false);
        }
    }
    catch (const cv::Exception& e)
    {
        throw FileException("Unable to create moment AVI: " + std::string(e.what()), false);
    }
}

size_t OutputMomentsAviFile::write_frame(const char* frame, size_t frame_size)
{
    const size_t pixels = fd_.get_frame_res();
    if (frame_size != pixels * movies_.size() * sizeof(float))
        throw FileException("Moment AVI export received an incomplete moment frame", false);

    const auto* values = reinterpret_cast<const float*>(frame);
    try
    {
        for (size_t moment = 0; moment < movies_.size(); ++moment)
        {
            auto& movie = movies_[moment];
            const bool signed_moment = moment == 1;
            if (movie.amplitude == 0.f)
            {
                for (size_t pixel = 0; pixel < pixels; ++pixel)
                {
                    const float value = values[pixel * movies_.size() + moment];
                    if (std::isfinite(value))
                        movie.amplitude = std::max(movie.amplitude, signed_moment ? std::abs(value) : value);
                }
            }

            const double offset = signed_moment ? 127.5 : 0.0;
            const double gain = movie.amplitude > 0.f ? (255.0 - offset) / movie.amplitude : 0.0;
            for (int y = 0; y < fd_.height; ++y)
            {
                // Wrap within the original image so the MJPEG padding stays at the edges.
                const int output_y = (y + shift_y_) % fd_.height;
                auto* row = image_.ptr<unsigned char>(output_y);
                int output_x = shift_x_;
                for (int x = 0; x < fd_.width; ++x)
                {
                    const size_t pixel = static_cast<size_t>(y) * fd_.width + x;
                    const float value = values[pixel * movies_.size() + moment];
                    if (!std::isfinite(value))
                        throw FileException("Non-finite values in moment M" + std::to_string(moment), false);
                    row[output_x] = cv::saturate_cast<unsigned char>(std::clamp(offset + gain * value, 0.0, 255.0));
                    if (++output_x == fd_.width)
                        output_x = 0;
                }
            }
            movie.writer.write(image_);
        }
    }
    catch (const cv::Exception& e)
    {
        throw FileException("Unable to write moment AVI frame: " + std::string(e.what()), false);
    }
    ++frames_written_;
    return frame_size;
}

void OutputMomentsAviFile::write_footer()
{
    if (frames_written_ == 0)
        throw FileException("No complete moment frames were recorded", false);

    for (auto& movie : movies_)
        movie.writer.release();
    for (auto& movie : movies_)
    {
        if (fs::exists(movie.path))
            throw FileException("Moment AVI path already exists: " + movie.path.string(), false);
        fs::rename(staging_dir_ / movie.path.filename(), movie.path);
        movie.published = true;
    }
    completed_ = true;
    for (const auto& movie : movies_)
        LOG_INFO("Wrote {}", movie.path.string());
}
} // namespace holovibes::io_files
