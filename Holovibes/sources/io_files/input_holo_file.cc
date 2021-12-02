#include <filesystem>
#include "input_holo_file.hh"
#include "file_exception.hh"
#include "compute_descriptor.hh"
#include "logger.hh"
#include "API.hh"

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

void import_holo_v4(holovibes::ComputeDescriptor& cd, const json& meta_data)
{
    api::json_to_compute_settings(meta_data["compute settings"]);

    const json& file_info_data = meta_data["file info"];
    cd.pixel_size = file_info_data["pixel size"]["x"];
    cd.raw_bitshift = file_info_data["raw bitshift"];
}

void import_holo_v2_v3(holovibes::ComputeDescriptor& cd, const json& meta_data)
{
    // This is done for retrocompatibility
    cd.compute_mode = get_value(meta_data, "mode", cd.compute_mode.load()) == Computation::Raw ? Computation::Raw
                                                                                               : Computation::Hologram;

    cd.space_transformation = get_value(meta_data, "algorithm", cd.space_transformation.load());
    cd.time_transformation = get_value(meta_data, "time_filter", cd.time_transformation.load());
    cd.time_transformation_size = get_value(meta_data, "#img", cd.time_transformation_size.load());
    cd.p.index = get_value(meta_data, "p", cd.p.index.load());
    cd.lambda = get_value(meta_data, "lambda", cd.lambda.load());
    cd.pixel_size = get_value(meta_data, "pixel_size", cd.pixel_size.load());
    cd.zdistance = get_value(meta_data, "z", cd.zdistance.load());
    cd.xy.log_scale_slice_enabled = get_value(meta_data, "log_scale", cd.xy.log_scale_slice_enabled.load());
    cd.xy.contrast_min = get_value(meta_data, "contrast_min", cd.xy.contrast_min.load());
    cd.xy.contrast_max = get_value(meta_data, "contrast_max", cd.xy.contrast_max.load());
    cd.fft_shift_enabled = get_value(meta_data, "fft_shift_enabled", cd.fft_shift_enabled.load());
    cd.x.accu_level = get_value(meta_data, "x_acc_level", cd.x.accu_level.load());
    cd.y.accu_level = get_value(meta_data, "y_acc_level", cd.y.accu_level.load());
    cd.p.accu_level = get_value(meta_data, "p_acc_level", cd.p.accu_level.load());
    cd.xy.img_accu_level = get_value(meta_data, "img_acc_slice_xy_level", cd.xy.img_accu_level.load());
    cd.xz.img_accu_level = get_value(meta_data, "img_acc_slice_xz_level", cd.xz.img_accu_level.load());
    cd.yz.img_accu_level = get_value(meta_data, "img_acc_slice_yz_level", cd.yz.img_accu_level.load());
    cd.renorm_enabled = get_value(meta_data, "renorm_enabled", cd.renorm_enabled.load());
}

void InputHoloFile::import_compute_settings(holovibes::ComputeDescriptor& cd) const
{
    if (holo_file_header_.version == 4)
        import_holo_v4(cd, meta_data_);
    else if (holo_file_header_.version < 4)
        import_holo_v2_v3(cd, meta_data_);
    else
        LOG_ERROR << "HOLO file version not supported!";
}
} // namespace holovibes::io_files
