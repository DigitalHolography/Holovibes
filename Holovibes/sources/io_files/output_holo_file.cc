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
    // export as a json
    try
    {
        Computation mode = Computation::Raw;

        if (record_raw && GSH::instance().get_compute_mode() == Computation::Hologram)
            mode = Computation::Hologram;

        meta_data_ = json{{"mode", mode},

                          {"algorithm", GSH::instance().get_space_transformation()},
                          {"time_filter", GSH::instance().get_time_transformation()},

                          {"#img", GSH::instance().get_time_transformation_size()},
                          {"p", GSH::instance().get_p_index()},
                          {"lambda", GSH::instance().get_lambda()},
                          {"pixel_size", cd.pixel_size.load()},
                          {"z", GSH::instance().get_z_distance()},

                          {"fft_shift_enabled", GSH::instance().get_fft_shift_enabled()},

                          {"x_acc_level", GSH::instance().get_x_accu_level()},
                          {"y_acc_level", GSH::instance().get_y_accu_level()},
                          {"p_acc_level", GSH::instance().get_p_accu_level()},

                          {"log_scale", GSH::instance().get_xy_log_scale_slice_enabled()},
                          {"contrast_min", GSH::instance().get_xy_contrast_min()},
                          {"contrast_max", GSH::instance().get_xy_contrast_max()},

                          {"img_acc_slice_xy_level", GSH::instance().get_xy_img_accu_level()},
                          {"img_acc_slice_xz_level", GSH::instance().get_xz_img_accu_level()},
                          {"img_acc_slice_yz_level", GSH::instance().get_yz_img_accu_level()},

                          {"renorm_enabled", cd.renorm_enabled.load()}};
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
