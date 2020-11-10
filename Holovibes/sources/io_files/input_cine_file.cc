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

namespace holovibes::io_files
{
    InputCineFile::InputCineFile(const std::string& file_path): InputFile(file_path), CineFile()
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

        frame_annotation_size_ = 8;
        actual_frame_size_ = fd_.frame_size() + frame_annotation_size_;
    }

    void InputCineFile::set_pos_to_first_frame()
    {
        // get the offset to the image offsets
        const std::fpos_t off_image_offset =
            static_cast<const std::fpos_t>(cine_file_header_.off_image_offset);

        std::fpos_t first_frame_offset = 0;

        // set pos to the image offsets, read the first frame offset and set pos to this offset
        if (std::fsetpos(file_, &off_image_offset) != 0
            || std::fread(&first_frame_offset, 1, sizeof(int64_t), file_) != sizeof(int64_t)
            || std::fsetpos(file_, &first_frame_offset) != 0)
        {
            throw FileException("Unable to seek the first frame");
        }
    }

    void InputCineFile::import_compute_settings(holovibes::ComputeDescriptor& cd) const
    {
        cd.pixel_size = 1e6 / static_cast<float>(bitmap_info_header_.bi_x_pels_per_meter);
    }
} // namespace holovibes::io_files
