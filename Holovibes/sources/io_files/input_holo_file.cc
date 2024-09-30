/*! \file
 *
 */

#include <filesystem>

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

    LOG_FUNC(file_path);

    InputHoloFile::load_header();

    InputHoloFile::load_fd();

    frame_size_ = fd_.get_frame_size();

    uintmax_t meta_data_size =
        std::filesystem::file_size(file_path) - (sizeof(HoloFileHeader) + holo_file_header_.total_data_size);

    has_footer = meta_data_size > 0 ? true : false;

    // perform a checksum
    if (holo_file_header_.total_data_size != frame_size_ * holo_file_header_.img_nb)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file: invalid total data size", false);
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
    LOG_FUNC();
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
        throw FileException("Invalid holo file: invalid bytecode size", false);
    }
    LOG_TRACE("Exiting InputHoloFile::load_header");
}

void InputHoloFile::load_fd()
{
    LOG_FUNC();
    fd_.width = holo_file_header_.img_width;
    fd_.height = holo_file_header_.img_height;
    fd_.depth = static_cast<camera::PixelDepth>(holo_file_header_.bits_per_pixel / 8);
    fd_.byteEndian = holo_file_header_.endianness ? camera::Endianness::BigEndian : camera::Endianness::LittleEndian;
    LOG_TRACE("Exiting InputHoloFile::load_fd");
}
void InputHoloFile::load_footer()
{
    LOG_FUNC();
    // compute the meta data offset to retrieve the meta data
    uintmax_t meta_data_offset = sizeof(HoloFileHeader) + holo_file_header_.total_data_size;
    uintmax_t file_size = std::filesystem::file_size(file_path_);

    if (meta_data_offset > file_size)
    {
        std::fclose(file_);
        throw FileException("Invalid holo file: invalid footer", false);
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
            LOG_WARN("An error occurred while retrieving the meta data. Meta data skipped");
        }
    }

    LOG_TRACE("Exiting InputHoloFile::load_footer");
}

void rec_fill_default_json(json& dst, json& src)
{
    for (auto dst_el = dst.begin(); dst_el != dst.end(); ++dst_el)
    {
        if (src.contains(dst_el.key()))
        {
            auto src_el = src.find(dst_el.key());
            if (dst_el->is_object())
            {
                rec_fill_default_json(*dst_el, *src_el);
            }
            else
            {
                dst.at(dst_el.key()) = src_el.value();
            }
        }
        // else : nothing to do, we keep the dst default json
    }
}

void InputHoloFile::import_compute_settings()
{
    LOG_FUNC();

    meta_data_ = json::parse("{}");
    // if there is no footer we use the state of the GSH
    if (!has_footer)
    {
        raw_footer_.Update();
        to_json(meta_data_, raw_footer_);
    }
    else
    {
        this->load_footer();
    }

    // perform convertion of holo file footer if needed
    if (holo_file_header_.version < 3)
        GSH::convert_json(meta_data_, GSH::ComputeSettingsVersion::V2);
    else if (holo_file_header_.version < 4)
        GSH::convert_json(meta_data_, GSH::ComputeSettingsVersion::V3);
    else if (holo_file_header_.version == 4)
        GSH::convert_json(meta_data_, GSH::ComputeSettingsVersion::V4);
    else if (holo_file_header_.version == 5)
        ;
    else
        LOG_ERROR("HOLO file version not supported!");

    if (!has_footer)
    {
        from_json(meta_data_, raw_footer_);
    }
    else
    {
        auto full_meta_data_ = json::parse("{}");
        raw_footer_.Update();
        to_json(full_meta_data_, raw_footer_);
        full_meta_data_["compute_settings"] = full_meta_data_;
        rec_fill_default_json(full_meta_data_, meta_data_);

        from_json(full_meta_data_["compute_settings"], raw_footer_);
    }

    // update GSH with the footer values
    raw_footer_.Assert();
    raw_footer_.Load();
}

void InputHoloFile::import_info() const
{
    LOG_FUNC();
    if (!has_footer)
        return;

    try
    {
        // Pixel are considered square
        api::set_pixel_size(meta_data_["info"]["pixel_pitch"]["x"]);
        api::set_input_fps(meta_data_["info"]["input_fps"]);
    }
    catch (std::exception&)
    {
    }
}

} // namespace holovibes::io_files