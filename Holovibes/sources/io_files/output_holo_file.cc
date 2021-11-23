#include "output_holo_file.hh"
#include "file_exception.hh"
#include "logger.hh"
#include "holovibes.hh"

namespace holovibes::io_files
{
OutputHoloFile::OutputHoloFile(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb)
    : OutputFrameFile(file_path)
    , HoloFile()
{
    fd_ = fd;

    holo_file_header_.magic_number[0] = 'H';
    holo_file_header_.magic_number[1] = 'O';
    holo_file_header_.magic_number[2] = 'L';
    holo_file_header_.magic_number[3] = 'O';

    holo_file_header_.version = current_version_;
    holo_file_header_.bits_per_pixel = fd_.depth * 8;
    holo_file_header_.img_width = fd_.width;
    holo_file_header_.img_height = fd_.height;
    holo_file_header_.img_nb = img_nb;
    holo_file_header_.endianness = camera::Endianness::LittleEndian;

    holo_file_header_.total_data_size = fd_.get_frame_size() * img_nb;

    meta_data_ = json();
}

void OutputHoloFile::export_compute_settings(bool record_raw)
{
    const auto& cd = ::holovibes::Holovibes::instance().get_cd();
    try
    {
        auto j_cs = json{
            {"image rendering",
             {
                 {"image mode", computation_to_string[cd.compute_mode.load()]},
                 {"batch size", cd.batch_size.load()},
                 {"time transformaton stride", cd.time_transformation_stride.load()},
                 {"filter2d",
                  {{"enabled", cd.filter2d_enabled.load()},
                   {"n1", cd.filter2d_n1.load()},
                   {"n2", cd.filter2d_n2.load()}}},
                 {"space tranformation", space_transformation_to_string[cd.space_transformation.load()]},
                 {"time transformation", time_transformation_to_string[cd.time_transformation.load()]},
                 {"time transformation size", cd.time_transformation_size.load()},
                 {"lambda", cd.lambda.load()},
                 {"z distance", cd.zdistance.load()},
                 {"convolution",
                  {{"enabled", cd.convolution_enabled.load()},
                   {"type", "45"},
                   {"divide", cd.divide_convolution_enabled.load()}}},
             }},
            {"view",
             {
                 {"type", "Magnitude"},
                 {"fft shift", cd.fft_shift_enabled.load()},
                 {"x", cd.x.to_json()},
                 {"y", cd.y.to_json()},
                 {"p", cd.p.to_json()},
                 {"q", cd.q.to_json()},
                 {"window",
                  {{"xy", cd.xy.to_json()},
                   {"yz", cd.yz.to_json()},
                   {"xz", cd.xz.to_json()},
                   {"filter2d", cd.filter2d.to_json()}}},
                 {"renorm", cd.renorm_enabled.load()},
                 {"reticle",
                  {{"display enabled", cd.reticle_display_enabled.load()}, {"scale", cd.reticle_scale.load()}}},
             }},
            {"composite",
             {
                 {"mode", composite_kind_to_string[cd.composite_kind.load()]},
                 {"auto weight", cd.composite_auto_weights.load()},
                 {"rgb", cd.rgb.to_json()},
                 {"hsv", cd.hsv.to_json()},
             }},
            {
                "advanced",
                {{"buffer size",
                  {{"input", cd.input_buffer_size.load()},
                   {"file", cd.file_buffer_size.load()},
                   {"record", cd.record_buffer_size.load()},
                   {"output", cd.output_buffer_size.load()},
                   {"time transformation cuts", cd.time_transformation_cuts_output_buffer_size.load()}}},
                 {
                     "filer2d smooth",
                     {{"low", cd.filter2d_smooth_low.load()}, {"high", cd.filter2d_smooth_high.load()}},
                 },
                 {"contrast",
                  {{"lower", cd.contrast_lower_threshold.load()},
                   {"upper", cd.contrast_upper_threshold.load()},
                   {"cuts p offset", cd.cuts_contrast_p_offset.load()}}},
                 {"renorm constant", cd.renorm_constant.load()}},
            },
        };

        auto j_fi = json{{"raw bitshift", cd.raw_bitshift.load()},
                         {"pixel size", {{"x", cd.pixel_size.load()}, {"y", cd.pixel_size.load()}}}};

        meta_data_ = json{{"compute_settings", j_cs}, {"file_info", j_fi}};
    }
    catch (const json::exception& e)
    {
        meta_data_ = json();
        LOG_WARN << "An error was encountered while trying to export compute settings";
        LOG_WARN << "Exception: " << e.what();
    }
}

void OutputHoloFile::write_header()
{
    if (std::fwrite(&holo_file_header_, 1, sizeof(HoloFileHeader), file_) != sizeof(HoloFileHeader))
        throw FileException("Unable to write output holo file header");
}

size_t OutputHoloFile::write_frame(const char* frame, size_t frame_size)
{
    size_t written_bytes = std::fwrite(frame, 1, frame_size, file_);

    if (written_bytes != frame_size)
        throw FileException("Unable to write output holo file frame");

    return written_bytes;
}

void OutputHoloFile::write_footer()
{
    const std::string& meta_data_str = meta_data_.dump();
    const size_t meta_data_size = meta_data_str.size();

    if (std::fwrite(meta_data_str.data(), 1, meta_data_size, file_) != meta_data_size)
        throw FileException("Unable to write output holo file footer");
}

void OutputHoloFile::correct_number_of_frames(size_t nb_frames_written)
{
    fpos_t previous_pos;

    if (std::fgetpos(file_, &previous_pos))
        throw FileException("Unable to correct number of written frames");

    holo_file_header_.img_nb = static_cast<uint32_t>(nb_frames_written);
    holo_file_header_.total_data_size = fd_.get_frame_size() * nb_frames_written;

    fpos_t file_begin_pos = 0;

    if (std::fsetpos(file_, &file_begin_pos))
        throw FileException("Unable to correct number of written frames");

    write_header();

    if (std::fsetpos(file_, &previous_pos))
        throw FileException("Unable to correct number of written frames");
}
} // namespace holovibes::io_files
