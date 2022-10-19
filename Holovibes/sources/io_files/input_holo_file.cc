#include <filesystem>
#include <iomanip>
#include <fstream>

#include "input_holo_file.hh"
#include "file_exception.hh"
#include "holovibes_config.hh"
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

void import_holo_v5(const json& meta_data)
{
    auto compute_settings = ComputeSettings();
    from_json(meta_data["compute_settings"], compute_settings);
    compute_settings.Load();
}

void InputHoloFile::import_compute_settings()
{
    LOG_FUNC(main);
    this->load_footer();
    if (holo_file_header_.version < 4)
    {
        apply_json_patch(meta_data_, "patch_v2-3_to_v5.json");
        meta_data_["compute_settings"]["image_rendering"]["space_transformation"] = static_cast<SpaceTransformation>(
            static_cast<int>(meta_data_["compute_settings"]["image_rendering"]["space_transformation"]));
        meta_data_["compute_settings"]["image_rendering"]["image_mode"] = static_cast<Computation>(
            static_cast<int>(meta_data_["compute_settings"]["image_rendering"]["image_mode"]) - 1);
        meta_data_["compute_settings"]["image_rendering"]["time_transformation"] = static_cast<TimeTransformation>(
            static_cast<int>(meta_data_["compute_settings"]["image_rendering"]["time_transformation"]));
    }
    else if (holo_file_header_.version == 4)
    {
        apply_json_patch(meta_data_, "patch_v4_to_v5.json");
    }
    else
    {
        LOG_ERROR(main, "HOLO file version not supported!");
    }
    import_holo_v5(meta_data_);
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

void InputHoloFile::apply_json_patch(json& meta_data, const std::string& json_patch_path)
{
    auto path_path = std::filesystem::path(holovibes::settings::patch_dirpath) / json_patch_path;
    auto file_content = std::ifstream(path_path, std::ifstream::in);
    auto patch = nlohmann::json::parse(file_content);

    meta_data = meta_data.patch(patch);
}

} // namespace holovibes::io_files
