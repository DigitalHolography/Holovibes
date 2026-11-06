#include "lsci_pca_cli.hh"
#include "lsci_pca.hh"
#include "input_frame_file_factory.hh"
#include "API.hh"
#include "logger.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <nlohmann/json.hpp>

namespace
{
// Same reduced, high-pass template-matching approach as live LSCI, but with a
// file-local reference and no GUI, compute workers or input-overflow queue.
class OfflineRegistration
{
    cv::Mat reference_, mask_;
    cv::Rect region_;
    double deviation_ = 0.;

  public:
    int dx = 0, dy = 0;
    cv::Mat align(const cv::Mat& map)
    {
        const double scale = std::min(1., 768. / std::max(map.cols, map.rows));
        cv::Mat features, background;
        cv::resize(map, features, cv::Size(), scale, scale, cv::INTER_AREA);
        const int side = std::min(features.cols, features.rows);
        if (side < 8)
            return map;
        cv::GaussianBlur(features, features, cv::Size(), std::max(.5, 3. * side / 432.));
        cv::GaussianBlur(features, background, cv::Size(), std::max(2., 24. * side / 432.));
        features -= background;
        const int search_radius = std::max(1, cvRound(.14 * side));
        if (reference_.empty())
        {
            const int radius = std::clamp(cvRound((side / 2 - search_radius - 1) * .95),
                                          2, side / 2 - search_radius - 1);
            region_ = cv::Rect(features.cols / 2 - radius, features.rows / 2 - radius, 2*radius+1, 2*radius+1);
            mask_.create(region_.size(), CV_32F);
            for (int y = -radius; y <= radius; ++y)
                for (int x = -radius; x <= radius; ++x)
                    mask_.at<float>(y+radius, x+radius) =
                        std::clamp(static_cast<float>((1. - std::hypot(x, y) / radius) / .15), 0.f, 1.f);
        }
        cv::Scalar mean, deviation;
        cv::meanStdDev(features(region_), mean, deviation, mask_ > 0);
        if (reference_.empty())
        {
            if (deviation[0] > 1e-6)
            {
                reference_ = features(region_).clone();
                deviation_ = deviation[0];
            }
            return map;
        }
        if (deviation[0] >= .15 * deviation_)
        {
            const cv::Rect search(region_.x-search_radius, region_.y-search_radius,
                                  region_.width+2*search_radius, region_.height+2*search_radius);
            cv::Mat correlation;
            cv::matchTemplate(features(search), reference_, correlation, cv::TM_CCOEFF_NORMED, mask_);
            for (int y = 0; y < correlation.rows; ++y)
                for (int x = 0; x < correlation.cols; ++x)
                    if (!std::isfinite(correlation.at<float>(y, x)))
                        correlation.at<float>(y, x) = -1.f;
            double score;
            cv::Point peak;
            cv::minMaxLoc(correlation, nullptr, &score, nullptr, &peak);
            if (score >= .55)
            {
                dx = -cvRound((peak.x-search_radius) * map.cols / static_cast<double>(features.cols));
                dy = -cvRound((peak.y-search_radius) * map.rows / static_cast<double>(features.rows));
            }
        }
        cv::Mat result;
        const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1, 0, dx, 0, 1, dy);
        cv::warpAffine(map, result, transform, map.size(), cv::INTER_NEAREST, cv::BORDER_WRAP);
        return result;
    }
};

std::pair<float, float> display_limits(const cv::Mat& map)
{
    std::vector<float> values;
    const double radius = .55 * std::min(map.rows, map.cols) / 2.;
    for (int y = 0; y < map.rows; ++y)
        for (int x = 0; x < map.cols; ++x)
            if (std::hypot(x-map.cols/2., y-map.rows/2.) <= radius)
                values.push_back(map.at<float>(y, x));
    if (values.empty())
        values.assign(map.ptr<float>(), map.ptr<float>() + map.total());
    const size_t low_index = static_cast<size_t>(.01 * (values.size()-1));
    const size_t high_index = static_cast<size_t>(.99 * (values.size()-1));
    std::nth_element(values.begin(), values.begin()+low_index, values.end());
    float low = values[low_index];
    std::nth_element(values.begin(), values.begin()+high_index, values.end());
    float high = values[high_index];
    if (high <= low + 1e-7f)
    {
        low -= 5e-7f;
        high = low + 1e-6f;
        LOG_WARN("First PCA-LSCI map has a degenerate display range; using a fixed 1e-6-wide range");
    }
    return {low, high};
}
}

namespace cli
{
int start_lsci_pca(const holovibes::OptionsDescriptor& opts)
{
    using namespace holovibes;
    const std::filesystem::path input_path(opts.input_path.value()), output_path(opts.output_path.value());
    if (input_path.extension() != ".holo" || output_path.extension() != ".avi")
        throw std::invalid_argument("Offline PCA-LSCI requires a .holo input and .avi output");
    if (opts.record_raw || opts.moments_record || opts.oct_cube_record || opts.oct_cube_float_record ||
        opts.frame_skip.value_or(0) != 0 || opts.compute_settings_path)
        throw std::invalid_argument("PCA-LSCI uses its own direct-intensity settings; raw/moments/OCT, frame skipping and -c are unavailable");
    const unsigned frames = opts.lsci_pca_frames, window = opts.lsci_pca_window;
    if (frames < 2 || frames > 512 || window < 1 || window > 63 || window % 2 == 0)
        throw std::invalid_argument("Use 2..512 PCA frames and an odd 1..63 spatial window");
    std::unique_ptr<io_files::InputFrameFile> input(io_files::InputFrameFileFactory::open(input_path.string()));
    const auto fd = input->get_frame_descriptor();
    if (API.input.get_data_type() != RecordedDataType::RAW ||
        (fd.depth != camera::PixelDepth::Bits8 && fd.depth != camera::PixelDepth::Bits16 && fd.depth != camera::PixelDepth::Bits32))
        throw std::invalid_argument("PCA-LSCI requires raw single-channel 8-bit, 16-bit or float32 intensities");
    if (fd.width % 2 || fd.height % 2)
        throw std::invalid_argument("PCA-LSCI AVI requires even image width and height for MJPEG encoding");
    const size_t available = input->get_total_nb_frames();
    const size_t start = opts.start_frame.value_or(1), end = opts.end_frame.value_or(static_cast<unsigned>(available));
    if (start < 1 || end > available || start > end)
        throw std::invalid_argument("Invalid PCA-LSCI start/end frame range");
    const size_t blocks = (end-start+1) / frames;
    const size_t count = opts.n_rec.value_or(static_cast<unsigned>(blocks));
    if (!count || count > blocks)
        throw std::invalid_argument("PCA-LSCI recording requires complete blocks; -n cannot exceed the available block count");
    const unsigned fps = opts.mp4_fps.value_or(24);
    if (!fps || fps > 24)
        throw std::invalid_argument("Offline PCA-LSCI AVI playback must be between 1 and 24 fps");

    compute::LsciPca processor(fd.width, fd.height, frames, window);
    cv::VideoWriter writer(output_path.string(), cv::VideoWriter::fourcc('M','J','P','G'),
                            fps, cv::Size(fd.width, fd.height), false);
    if (!writer.isOpened())
        throw std::runtime_error("Unable to create PCA-LSCI AVI: " + output_path.string());
    LOG_INFO("Offline PCA-LSCI: {} frames/block, {}x{} spatial window, {} blocks, {} fps AVI, registration {}",
             frames, window, window, count, fps, !opts.lsci_pca_no_registration);
    const size_t tail = (end-start+1) % frames;
    if (tail)
        LOG_INFO("PCA-LSCI ignores {} trailing input frames (incomplete block)", tail);
    nlohmann::json metadata = {
        {"input", input_path.string()}, {"algorithm", "PCA static reconstruction then K(original)-K(static)"},
        {"centering", "spatial mean of each frame"}, {"static_selection", "eigenvalue >= mean of N eigenvalues"},
        {"frames_per_block", frames}, {"spatial_window", window}, {"width", fd.width}, {"height", fd.height},
        {"playback_fps", fps}, {"start_frame", start}, {"end_frame", end}, {"output_frames", count},
        {"trailing_incomplete_frames", tail}, {"registration_enabled", !opts.lsci_pca_no_registration},
        {"display_percentiles", {1, 99}}, {"display_circle_scale", .55}, {"display_range_fixed", true},
        {"blocks", nlohmann::json::array()}
    };
    if (opts.fps)
        metadata["input_fps"] = opts.fps.value(); // Provenance only: offline reading is not paced in real time.
    std::vector<unsigned char> frame(fd.get_frame_size());
    input->set_pos_to_frame(start-1);
    OfflineRegistration registration;
    float low = 0, high = 0;
    for (size_t block = 0; block < count; ++block)
    {
        for (unsigned t = 0; t < frames; ++t)
        {
            int packed;
            if (input->read_frames(reinterpret_cast<char*>(frame.data()), 1, &packed) != 1)
                throw std::runtime_error("Unexpected end of PCA-LSCI input");
            if (fd.byteEndian == camera::Endianness::BigEndian && fd.depth > 1)
                for (size_t p = 0; p < frame.size(); p += fd.depth)
                    std::reverse(frame.begin()+p, frame.begin()+p+fd.depth);
            processor.add_frame(frame.data(), fd.depth);
        }
        auto result = processor.process();
        cv::Mat dynamic(fd.height, fd.width, CV_32F, result.dynamic.data());
        cv::Mat registered = opts.lsci_pca_no_registration ? dynamic : registration.align(dynamic);
        if (block == 0)
        {
            std::tie(low, high) = display_limits(registered);
            metadata["display_K_difference_min"] = low;
            metadata["display_K_difference_max"] = high;
            LOG_INFO("PCA-LSCI display limits fixed from first map: [{}, {}]", low, high);
        }
        cv::Mat display;
        registered.convertTo(display, CV_8U, 255. / (high-low), -255. * low / (high-low));
        writer.write(display);
        metadata["blocks"].push_back({{"input_first_frame", start+block*frames},
            {"static_components", result.static_components}, {"eigenvalues", result.eigenvalues},
            {"registration_dx", registration.dx}, {"registration_dy", registration.dy}});
        LOG_INFO("PCA-LSCI block {}/{}: {} static components", block+1, count, result.static_components);
    }
    writer.release();
    std::ofstream sidecar(output_path.string() + ".json");
    sidecar << metadata.dump(2) << '\n';
    sidecar.close();
    if (!sidecar)
        throw std::runtime_error("Unable to save PCA-LSCI processing metadata");
    LOG_INFO("Offline PCA-LSCI complete: {} maps saved to {}", count, output_path.string());
    return 0;
}
}
