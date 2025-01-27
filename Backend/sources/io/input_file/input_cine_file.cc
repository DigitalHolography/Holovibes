/*! \file
 *
 */

#include "input_cine_file.hh"
#include "file_exception.hh"

#include "API.hh"

namespace holovibes::io_files
{
InputCineFile::InputCineFile(const std::string& file_path)
    : InputFrameFile(file_path)
    , CineFile()
{

    has_footer = false;

    // read the cine file and bitmap info headers
    size_t bytes_read = std::fread(&cine_file_header_, sizeof(char), sizeof(CineFileHeader), file_);
    bytes_read += std::fread(&bitmap_info_header_, sizeof(char), sizeof(BitmapInfoHeader), file_);

    if (std::ferror(file_))
    {
        std::fclose(file_);
        throw FileException("An error was encountered while reading the file");
    }

    // if the data has not been fully retrieved or the cine file is not an
    // actual cine file
    if (bytes_read != sizeof(CineFileHeader) + sizeof(BitmapInfoHeader) ||
        strncmp("CI", reinterpret_cast<char*>(&cine_file_header_.type), 2) != 0)
    {
        std::fclose(file_);
        throw FileException("Invalid cine file", false);
    }

    fd_.width = std::abs(bitmap_info_header_.bi_width);
    fd_.height = std::abs(bitmap_info_header_.bi_height);
    fd_.depth = static_cast<camera::PixelDepth>(bitmap_info_header_.bi_bit_count / 8);
    fd_.byteEndian = camera::Endianness::LittleEndian;

    frame_size_ = fd_.get_frame_size();
    packed_frame_size_ = bitmap_info_header_.bi_size_image;
}

json InputCineFile::import_compute_settings() { return {}; }

void InputCineFile::import_info() const
{
    float px = static_cast<float>(bitmap_info_header_.bi_x_pels_per_meter);
    API.input.set_pixel_size(1e6f / px);
}

void InputCineFile::set_pos_to_frame(size_t frame_id)
{
    // get the offset to the frame offset
    const std::fpos_t offset_frame_offset =
        static_cast<const std::fpos_t>(cine_file_header_.off_image_offset + frame_id * sizeof(int64_t));

    std::fpos_t frame_offset = 0;

    // set pos to the image offsets, read the first frame offset and set pos to
    // this offset
    if (std::fsetpos(file_, &offset_frame_offset) != 0 ||
        std::fread(&frame_offset, 1, sizeof(int64_t), file_) != sizeof(int64_t) ||
        std::fsetpos(file_, &frame_offset) != 0)
    {
        throw FileException("Unable to seek the frame requested");
    }
}

size_t InputCineFile::read_frames(char* buffer, size_t frames_to_read, int* flag_packed)
{
    size_t frames_read = 0;

    *flag_packed = static_cast<int>((packed_frame_size_ / (float)(fd_.width * fd_.height)) * 8);

    for (size_t i = 0; i < frames_to_read; i++)
    {
        // skip annotation before each frame
        // FIXME: seek was hardcoded in the previous versions of Holovibes
        // This eventually matches the recorded cine but will be invalid for
        // other cine files
        if (std::fseek(file_, 8, SEEK_CUR) != 0)
            throw FileException("Unable to read " + std::to_string(frames_to_read) + " frames");

        frames_read += std::fread(buffer + i * packed_frame_size_, packed_frame_size_, 1, file_);

        if (ferror(file_))
            throw FileException("Unable to read " + std::to_string(frames_to_read) + " frames");
    }

    return frames_read;
}

} // namespace holovibes::io_files
