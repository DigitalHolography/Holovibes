#include <filesystem>
#include <iomanip>
#include "input_holo_file.hh"
#include "file_exception.hh"

#include "logger.hh"
#include "all_struct.hh"
#include "API.hh"
#include "global_state_holder.hh"

#include "internals_struct.hh"
#include "compute_settings_struct.hh"

namespace holovibes::io_files
{
InputHoloFile::InputHoloFile(const std::string& file_path)
    : InputFrameFile(file_path)
    , HoloFile()
{

    // fd_.width = holo_file_header_.img_width;
    // fd_.height = holo_file_header_.img_height;
    // fd_.depth = holo_file_header_.bits_per_pixel / 8;
    // fd_.byteEndian = holo_file_header_.endianness ? camera::Endianness::BigEndian : camera::Endianness::LittleEndian;
    LOG_FUNC(main, file_path);

    InputHoloFile::load_header();

    InputHoloFile::load_fd();

    frame_size_ = fd_.get_frame_size();

    // perform a checksum
    if (holo_file_header_.total_data_size != frame_size_ * holo_file_header_.img_nb)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file", false);
    }
}

void InputHoloFile::set_pos_to_frame(size_t frame_id)
{
    std::fpos_t frame_offset = sizeof(HoloFileHeader) + frame_size_ * frame_id;

    if (std::fsetpos(file_, &frame_offset) != 0)
        throw FileException("Unable to seek the frame requested");
}

void InputHoloFile::load_header()
{
    LOG_FUNC(main);
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
    LOG_TRACE(main, "Exiting InputHoloFile::load_header");
}

void InputHoloFile::load_fd()
{
    LOG_FUNC(main);
    fd_.width = holo_file_header_.img_width;
    fd_.height = holo_file_header_.img_height;
    fd_.depth = holo_file_header_.bits_per_pixel / 8;
    fd_.byteEndian = holo_file_header_.endianness ? camera::Endianness::BigEndian : camera::Endianness::LittleEndian;
    LOG_TRACE(main, "Exiting InputHoloFile::load_fd");
}
void InputHoloFile::load_footer()
{
    LOG_FUNC(main);
    // compute the meta data offset to retrieve the meta data
    uintmax_t meta_data_offset = sizeof(HoloFileHeader) + holo_file_header_.total_data_size;
    uintmax_t file_size = std::filesystem::file_size(file_path_);

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
            LOG_WARN(main, "An error occurred while retrieving the meta data. Meta data skipped");
        }
    }
    LOG_TRACE(main, "Exiting InputHoloFile::load_footer");
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

template <typename T>
void convert_value(json& json, const std::string& key_json, const T& default_value)
{
    if (!json.contains(key) || json[key].is_null())
    {
        json[key] = default_value;
    }
}

void import_holo_v4(const json& meta_data)
{
    if (meta_data.contains("compute settings"))
        api::json_to_compute_settings(meta_data["compute settings"]);
}

void import_holo_v5(const json& meta_data)
{
        api::json_to_compute_settings_v5(meta_data);
}

// This is done for retrocompatibility
void import_holo_v2_v3(const json& meta_data)
{
    GSH::instance().set_space_transformation(
        get_value(meta_data, "algorithm", GSH::instance().get_space_transformation()));
    GSH::instance().set_time_transformation(
        get_value(meta_data, "time_filter", GSH::instance().get_time_transformation()));
    GSH::instance().set_time_transformation_size(
        get_value(meta_data, "#img", GSH::instance().get_time_transformation_size()));
    GSH::instance().set_p_index(get_value(meta_data, "p", GSH::instance().get_p_index()));
    GSH::instance().set_lambda(get_value(meta_data, "lambda", GSH::instance().get_lambda()));
    GSH::instance().set_z_distance(get_value(meta_data, "z", GSH::instance().get_z_distance()));
    GSH::instance().set_xy_log_scale_slice_enabled(
        get_value(meta_data, "log_scale", GSH::instance().get_xy_log_scale_slice_enabled()));
    GSH::instance().set_xy_contrast_min(get_value(meta_data, "contrast_min", GSH::instance().get_xy_contrast_min()));
    GSH::instance().set_xy_contrast_max(get_value(meta_data, "contrast_max", GSH::instance().get_xy_contrast_max()));
    GSH::instance().set_x_accu_level(get_value(meta_data, "x_acc_level", GSH::instance().get_x_accu_level()));
    GSH::instance().set_y_accu_level(get_value(meta_data, "y_acc_level", GSH::instance().get_y_accu_level()));
    GSH::instance().set_p_accu_level(get_value(meta_data, "p_acc_level", GSH::instance().get_p_accu_level()));
    GSH::instance().set_xy_img_accu_level(
        get_value(meta_data, "img_acc_slice_xy_level", GSH::instance().get_xy_img_accu_level()));
    GSH::instance().set_xz_img_accu_level(
        get_value(meta_data, "img_acc_slice_xz_level", GSH::instance().get_xz_img_accu_level()));
    GSH::instance().set_yz_img_accu_level(
        get_value(meta_data, "img_acc_slice_yz_level", GSH::instance().get_yz_img_accu_level()));

    if (meta_data.contains("mode"))
    {
        GSH::instance().set_compute_mode(meta_data["mode"]);
        GSH::instance().set_compute_mode(
            static_cast<Computation>(static_cast<int>(GSH::instance().get_compute_mode()) - 1));
    }

    GSH::instance().set_fft_shift_enabled(
        get_value(meta_data, "fft_shift_enabled", GSH::instance().get_fft_shift_enabled()));
    GSH::instance().set_renorm_enabled(get_value(meta_data, "renorm_enabled", GSH::instance().get_renorm_enabled()));
}

void InputHoloFile::import_compute_settings()
{
    LOG_FUNC(main);
    if (holo_file_header_.version < 4)
    {
        convert_holo_footer_to_v4(meta_data_);
        import_holo_v5(meta_data_);
        // import_holo_v4(meta_data_);
        //  import_holo_v2_v3(meta_data_);
    }
    else if (holo_file_header_.version == 4)
        import_holo_v4(meta_data_);
    else
    {
        LOG_ERROR(main, "HOLO file version not supported!");
    }
}

void InputHoloFile::import_info() const
{
    if (holo_file_header_.version == 4)
    {
        if (meta_data_.contains("info"))
        {
            const json& file_info_data = meta_data_["info"];
            GSH::instance().set_raw_bitshift(
                get_value(file_info_data, "raw bitshift", GSH::instance().get_raw_bitshift()));

            if (file_info_data.contains("pixel size"))
            {
                const json& pixel_size_data = file_info_data["pixel size"];
                GSH::instance().set_pixel_size(get_value(pixel_size_data, "x", GSH::instance().get_pixel_size()));
            }
        }
    }
    else if (holo_file_header_.version < 4)
    {
        GSH::instance().set_pixel_size(get_value(meta_data_, "pixel_size", GSH::instance().get_pixel_size()));
    }
    else
    {
        LOG_ERROR(main, "HOLO file version not supported!");
    }
}

void InputHoloFile::convert_holo_footer_to_v4(json& meta_data)
{

    raw_footer_.image_rendering.time_transformation_size = meta_data["#img"];
    raw_footer_.image_rendering.space_transformation = static_cast<SpaceTransformation>(static_cast<int>(meta_data["algorithm"]));

    raw_footer_.view.window.xy.contrast.max = meta_data["contrast_max"];
    raw_footer_.view.window.xy.contrast.min = meta_data["contrast_min"];

    raw_footer_.view.fft_shift = meta_data["fft_shift_enabled"];
    raw_footer_.view.window.xy.img_accu_level = meta_data["img_acc_slice_xy_level"];
    raw_footer_.view.window.xz.img_accu_level = meta_data["img_acc_slice_xz_level"];
    raw_footer_.view.window.yz.img_accu_level = meta_data["img_acc_slice_yz_level"];

    raw_footer_.image_rendering.lambda = meta_data["lambda"];
    raw_footer_.view.window.xy.log_enabled = meta_data["log_scale"];

    if (meta_data.contains("mode"))
    {
        raw_footer_.image_rendering.image_mode = static_cast<Computation>(static_cast<int>(meta_data["mode"]) - 1);
    }
    raw_footer_.view.p.index = meta_data["p"];

    raw_footer_.view.p.accu_level = meta_data["p_acc_level"];

    raw_footer_.view.renorm = meta_data["renorm_enabled"];
    raw_footer_.image_rendering.time_transformation = static_cast<TimeTransformation>(static_cast<int>(meta_data["time_filter"]));

    raw_footer_.view.x.accu_level = meta_data["x_acc_level"];
    raw_footer_.view.y.accu_level = meta_data["y_acc_level"];
    raw_footer_.image_rendering.z_distance = meta_data["z"];

    raw_footer_.image_rendering.convolution.type = "None";

    meta_data.clear();
    to_json(meta_data, raw_footer_);
}

} // namespace holovibes::io_files
