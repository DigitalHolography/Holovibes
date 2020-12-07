/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "input_cine_file.hh"
#include "file_exception.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
    InputCineFile::InputCineFile(const std::string& file_path): InputFrameFile(file_path), CineFile()
    {
        // read the cine file and bitmap info headers
        size_t bytes_read = std::fread(&cine_file_header_, sizeof(char), sizeof(CineFileHeader), file_);
        bytes_read += std::fread(&bitmap_info_header_, sizeof(char), sizeof(BitmapInfoHeader), file_);

        if (std::ferror(file_))
        {
            std::fclose(file_);
            throw FileException("An error was encountered while reading the file");
        }

        // if the data has not been fully retrieved or the cine file is not an actual cine file
        if (bytes_read != sizeof(CineFileHeader) + sizeof(BitmapInfoHeader)
            || strncmp("CI", reinterpret_cast<char*>(&cine_file_header_.type), 2) != 0)
        {
            std::fclose(file_);
            throw FileException("Invalid cine file");
        }

        fd_.width = std::abs(bitmap_info_header_.bi_width);
        fd_.height = std::abs(bitmap_info_header_.bi_height);
        fd_.depth = bitmap_info_header_.bi_bit_count / 8;
        fd_.byteEndian = camera::Endianness::LittleEndian;

        frame_size_ = fd_.frame_size();
    }

    void InputCineFile::import_compute_settings(holovibes::ComputeDescriptor& cd) const
    {
        cd.pixel_size = 1e6 / static_cast<float>(bitmap_info_header_.bi_x_pels_per_meter);
    }

    void InputCineFile::set_pos_to_frame(size_t frame_id)
    {
        // get the offset to the frame offset
        const std::fpos_t offset_frame_offset =
            static_cast<const std::fpos_t>(cine_file_header_.off_image_offset + frame_id * sizeof(int64_t));

        std::fpos_t frame_offset = 0;

        // set pos to the image offsets, read the first frame offset and set pos to this offset
        if (std::fsetpos(file_, &offset_frame_offset) != 0
            || std::fread(&frame_offset, 1, sizeof(int64_t), file_) != sizeof(int64_t)
            || std::fsetpos(file_, &frame_offset) != 0)
        {
            throw FileException("Unable to seek the frame requested");
        }
    }

    size_t InputCineFile::read_frames(char* buffer, size_t frames_to_read)
    {
        size_t frames_read = 0;

        for (size_t i = 0; i < frames_to_read; i++)
        {
            // skip annotation before each frame
            // FIXME: seek was hardcoded in the previous versions of Holovibes
            // This eventually matches the recorded cine but will be invalid for other cine files
            if (std::fseek(file_, 8, SEEK_CUR) != 0)
                throw FileException("Unable to read " + std::to_string(frames_to_read) + " frames");

            frames_read += std::fread(buffer + i * frame_size_, frame_size_, 1, file_);

            if (ferror(file_))
                throw FileException("Unable to read " + std::to_string(frames_to_read) + " frames");
        }

        return frames_read;
    }
} // namespace holovibes::io_files
