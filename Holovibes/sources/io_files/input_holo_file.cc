#include <filesystem>
#include "input_holo_file.hh"
#include "file_exception.hh"
#include "compute_descriptor.hh"
#include "logger.hh"
#include "global_state_holder.hh"

namespace holovibes::io_files
{
InputHoloFile::InputHoloFile(const std::string& file_path)
    : InputFrameFile(file_path)
    , HoloFile()
{
    // read the file header
    size_t bytes_read = std::fread(&holo_file_header_, sizeof(char), sizeof(HoloFileHeader), file_);

    if (std::ferror(file_))
    {
        std::fclose(file_);
        throw FileException("An error was encountered while reading the file");
    }

    // if the data has not been fully retrieved or the holo file is not an
    // actual holo file
    if (bytes_read != sizeof(HoloFileHeader) || std::strncmp("HOLO", holo_file_header_.magic_number, 4) != 0)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file", false);
    }

    fd_.width = holo_file_header_.img_width;
    fd_.height = holo_file_header_.img_height;
    fd_.depth = holo_file_header_.bits_per_pixel / 8;
    fd_.byteEndian = holo_file_header_.endianness ? camera::Endianness::BigEndian : camera::Endianness::LittleEndian;

    frame_size_ = fd_.get_frame_size();

    // perform a checksum
    if (holo_file_header_.total_data_size != frame_size_ * holo_file_header_.img_nb)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file", false);
    }

    // compute the meta data offset to retrieve the meta data
    uintmax_t meta_data_offset = sizeof(HoloFileHeader) + holo_file_header_.total_data_size;
    uintmax_t file_size = std::filesystem::file_size(file_path);

    if (meta_data_offset > file_size)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file", false);
    }

    uintmax_t meta_data_size = file_size - meta_data_offset;

    // retrieve the meta data
    meta_data_ = json::parse("{}");
    if (meta_data_size > 0)
    {
        std::string meta_data_str;

        // handle crash if meta_data_size is greater than max_size()
        try
        {
            meta_data_str.resize(meta_data_size + 1);
            meta_data_str[meta_data_size] = 0;

            if (std::fsetpos(file_, reinterpret_cast<std::fpos_t*>(&meta_data_offset)) == 0 &&
                std::fread(meta_data_str.data(), sizeof(char), meta_data_size, file_) == meta_data_size)
            {
                meta_data_ = json::parse(meta_data_str);
            }
        }
        catch (const std::exception&)
        {
            // does not throw an error if the meta data are not parsed
            // because they are not essential
            LOG_WARN << "An error occurred while retrieving the meta data. Meta "
                     << "data skipped";
        }
    }
}

void InputHoloFile::set_pos_to_frame(size_t frame_id)
{
    std::fpos_t frame_offset = sizeof(HoloFileHeader) + frame_size_ * frame_id;

    if (std::fsetpos(file_, &frame_offset) != 0)
        throw FileException("Unable to seek the frame requested");
}

template <typename T>
T get_value(const json& json, const std::string& key, const T& default_value)
{
    if (!json.contains(key) || json[key].is_null())
    {
        return default_value;
    }
    return json[key];
}

void InputHoloFile::import_compute_settings(holovibes::ComputeDescriptor& cd) const
{
    GSH::instance().set_space_transformation(
        get_value(meta_data_, "algorithm", GSH::instance().get_space_transformation()));
    GSH::instance().set_time_transformation(
        get_value(meta_data_, "time_filter", GSH::instance().get_time_transformation()));
    GSH::instance().set_time_transformation_size(
        get_value(meta_data_, "#img", GSH::instance().get_time_transformation_size()));
    GSH::instance().set_p_index(get_value(meta_data_, "p", GSH::instance().get_p_index()));
    GSH::instance().set_lambda(get_value(meta_data_, "lambda", GSH::instance().get_lambda()));
    GSH::instance().set_z_distance(get_value(meta_data_, "z", GSH::instance().get_z_distance()));
    GSH::instance().set_xy_log_scale_slice_enabled(
        get_value(meta_data_, "log_scale", GSH::instance().get_xy_log_scale_slice_enabled()));
    GSH::instance().set_xy_contrast_min(get_value(meta_data_, "contrast_min", GSH::instance().get_xy_contrast_min()));
    GSH::instance().set_xy_contrast_max(get_value(meta_data_, "contrast_max", GSH::instance().get_xy_contrast_max()));
    GSH::instance().set_x_accu_level(get_value(meta_data_, "x_acc_level", GSH::instance().get_x_accu_level()));
    GSH::instance().set_y_accu_level(get_value(meta_data_, "y_acc_level", GSH::instance().get_y_accu_level()));
    // cd.p.accu_level = get_value(meta_data_, "p_acc_level", cd.p.accu_level);
    GSH::instance().set_p_accu_level(get_value(meta_data_, "p_acc_level", GSH::instance().get_p_accu_level()));
    GSH::instance().set_xy_img_accu_level(
        get_value(meta_data_, "img_acc_slice_xy_level", GSH::instance().get_xy_img_accu_level()));
    GSH::instance().set_xz_img_accu_level(
        get_value(meta_data_, "img_acc_slice_xz_level", GSH::instance().get_xz_img_accu_level()));
    GSH::instance().set_yz_img_accu_level(
        get_value(meta_data_, "img_acc_slice_yz_level", GSH::instance().get_yz_img_accu_level()));

    cd.compute_mode = get_value(meta_data_, "mode", cd.compute_mode.load());
    cd.pixel_size = get_value(meta_data_, "pixel_size", cd.pixel_size.load());
    cd.fft_shift_enabled = get_value(meta_data_, "fft_shift_enabled", cd.fft_shift_enabled.load());
    cd.renorm_enabled = get_value(meta_data_, "renorm_enabled", cd.renorm_enabled.load());
}
} // namespace holovibes::io_files
