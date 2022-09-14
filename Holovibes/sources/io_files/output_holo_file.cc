#include "output_holo_file.hh"
#include "file_exception.hh"
#include "logger.hh"
#include "holovibes.hh"
#include "API.hh"

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

void OutputHoloFile::export_compute_settings(int input_fps, size_t contiguous)
{
    Logger::main().trace("Entering OutputHoloFile export_compute_settings()");
    Logger::main().debug("raw bitshift : {}", GSH::instance().get_raw_bitshift());

    try
    {
        auto j_fi =
            json{{"raw bitshift", GSH::instance().get_raw_bitshift()},
                 {"pixel size", {{"x", GSH::instance().get_pixel_size()}, {"y", GSH::instance().get_pixel_size()}}},
                 {"input fps", input_fps},
                 {"contiguous", contiguous}};
        meta_data_ = json{{"compute settings", holovibes::api::compute_settings_to_json()}, {"info", j_fi}};
    }
    catch (const json::exception& e)
    {
        meta_data_ = json();
        Logger::main().warn("An error was encountered while trying to export compute settings");
        Logger::main().warn("Exception: {}", e.what());
    }
}

void OutputHoloFile::write_header()
{
    if (std::fwrite(&holo_file_header_, 1, sizeof(HoloFileHeader), file_) != sizeof(HoloFileHeader))
        throw FileException("Unable to write output holo file header");
}

size_t OutputHoloFile::write_frame(const char* frame, size_t frame_size)
{
    const size_t written_bytes = std::fwrite(frame, 1, frame_size, file_);

    if (written_bytes != frame_size)
        throw FileException("Unable to write output holo file frame");

    return written_bytes;
}

void OutputHoloFile::write_footer()
{
    Logger::main().trace("Entering OutputHoloFile::write_footer()");
    std::string meta_data_str;
    try
    {
        meta_data_str = meta_data_.dump();
        if (std::fwrite(meta_data_str.data(), 1, meta_data_str.size(), file_) != meta_data_str.size())
            throw FileException("Unable to write output holo file footer");
    }
    catch (const std::exception& e)
    {
        Logger::main().error("{}", e.what());
        throw;
    }
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
