#include "output_holo_file.hh"
#include "file_exception.hh"
#include "logger.hh"

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

    holo_file_header_.total_data_size = fd_.frame_size() * img_nb;

    meta_data_ = json();
}

void OutputHoloFile::export_compute_settings(const ComputeDescriptor& cd, bool record_raw)
{
    // export as a json
    try
    {
        Computation mode = Computation::Raw;

        if (record_raw && cd.compute_mode.load() == Computation::Hologram)
            mode = Computation::Hologram;

        meta_data_ = json{{"mode", mode},

                          {"algorithm", cd.space_transformation.load()},
                          {"time_filter", cd.time_transformation.load()},

                          {"#img", cd.time_transformation_size.load()},
                          {"p", cd.p.index.load()},
                          {"lambda", cd.lambda.load()},
                          {"pixel_size", cd.pixel_size.load()},
                          {"z", cd.zdistance.load()},

                          {"fft_shift_enabled", cd.fft_shift_enabled.load()},

                          {"x_acc_enabled", cd.x.accu_enabled.load()},
                          {"x_acc_level", cd.x.accu_level.load()},
                          {"y_acc_enabled", cd.y.accu_enabled.load()},
                          {"y_acc_level", cd.y.accu_level.load()},
                          {"p_acc_enabled", cd.p.accu_enabled.load()},
                          {"p_acc_level", cd.p.accu_level.load()},

                          {"log_scale", cd.xy.log_scale_slice_enabled.load()},
                          {"contrast_min", cd.contrast_min_slice_xy.load()},
                          {"contrast_max", cd.contrast_max_slice_xy.load()},

                          {"img_acc_slice_xy_enabled", cd.xy.img_acc_slice_enabled.load()},
                          {"img_acc_slice_xz_enabled", cd.xz.img_acc_slice_enabled.load()},
                          {"img_acc_slice_yz_enabled", cd.yz.img_acc_slice_enabled.load()},
                          {"img_acc_slice_xy_level", cd.xy.img_acc_slice_level.load()},
                          {"img_acc_slice_xz_level", cd.xz.img_acc_slice_level.load()},
                          {"img_acc_slice_yz_level", cd.yz.img_acc_slice_level.load()},

                          {"renorm_enabled", cd.renorm_enabled.load()}};
    }
    catch (const json::exception& e)
    {
        meta_data_ = json();
        LOG_WARN << "An error was encountered while trying to export compute settings";
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

    holo_file_header_.img_nb = nb_frames_written;
    holo_file_header_.total_data_size = fd_.frame_size() * nb_frames_written;

    fpos_t file_begin_pos = 0;

    if (std::fsetpos(file_, &file_begin_pos))
        throw FileException("Unable to correct number of written frames");

    write_header();

    if (std::fsetpos(file_, &previous_pos))
        throw FileException("Unable to correct number of written frames");
}
} // namespace holovibes::io_files
