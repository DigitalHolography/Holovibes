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

#include <filesystem>
#include "input_holo_file.hh"
#include "file_exception.hh"
#include "logger.hh"

namespace holovibes::io_files
{
    InputHoloFile::InputHoloFile(const std::string& file_path): InputFile(file_path), HoloFile()
    {
		// read the file header
        size_t bytes_read = std::fread(&holo_file_header_, sizeof(char), sizeof(HoloFileHeader), file_);

        if (std::ferror(file_))
        {
            std::fclose(file_);
            throw FileException("An error was encountered while reading the file");
        }

        // if the data has not been fully retrieved or the holo file is not an actual holo file
		if (bytes_read != sizeof(HoloFileHeader)
            || std::strncmp("HOLO", holo_file_header_.magic_number, 4) != 0)
		{
            std::fclose(file_);
            throw FileException("Invalid holo file");
		}

		fd_.width = holo_file_header_.img_width;
		fd_.height = holo_file_header_.img_height;
		fd_.depth = holo_file_header_.bits_per_pixel / 8;
		fd_.byteEndian = holo_file_header_.endianness ?
                        camera::Endianness::BigEndian : camera::Endianness::LittleEndian;

        size_t frame_size = fd_.frame_size();
        frame_annotation_size_ = 0;
        actual_frame_size_ = frame_size + frame_annotation_size_;

        // perform a checksum
        if (holo_file_header_.total_data_size != frame_size * holo_file_header_.img_nb)
            throw FileException("Invalid holo file");

		// compute the meta data offset to retrieve the meta data
		uintmax_t meta_data_offset = sizeof(HoloFileHeader) + holo_file_header_.total_data_size;
		uintmax_t file_size = std::filesystem::file_size(file_path);

        if (meta_data_offset > file_size)
        {
            std::fclose(file_);
            throw FileException("Invalid holo file");
        }

		uintmax_t meta_data_size = file_size - meta_data_offset;

		// retrieve the meta data
		if (meta_data_size > 0)
		{
            std::string meta_data_str;

			// handle crash if meta_data_size is greater than max_size()
			try
			{
				meta_data_str.resize(meta_data_size + 1);
                meta_data_str[meta_data_size] = 0;

                if (std::fsetpos(file_, reinterpret_cast<std::fpos_t*>(&meta_data_offset)) == 0
                    && std::fread(meta_data_str.data(), sizeof(char), meta_data_size, file_) == meta_data_size)
                {
                    meta_data_ = json::parse(meta_data_str);
                }
			}
			catch (const std::exception&)
			{
                // does not throw an error if the meta data are not parsed
                // because they are not essential
                LOG_WARN("An error occurred while retrieving the meta data. Meta data skipped");
                meta_data_ = json();
            }
		}
    }

    void InputHoloFile::set_pos_to_first_frame()
    {
        std::fpos_t first_frame_offset = sizeof(HoloFileHeader);

        if (std::fsetpos(file_, &first_frame_offset) != 0)
            throw FileException("Unable to seek the first frame");
    }

    void InputHoloFile::import_compute_settings(holovibes::ComputeDescriptor& cd) const
    {
        cd.compute_mode = meta_data_.value("mode", Computation::Raw);
        cd.algorithm = meta_data_.value("algorithm", Algorithm::None);
        cd.time_filter = meta_data_.value("time_filter", TimeFilter::STFT);
        cd.time_filter_size = meta_data_.value("#img", 1);
        cd.pindex = meta_data_.value("p", 0);
        cd.lambda = meta_data_.value("lambda", 0.0f);
        cd.pixel_size = meta_data_.value("pixel_size", 12.0);
        cd.zdistance = meta_data_.value("z", 0.0f);
        cd.log_scale_slice_xy_enabled = meta_data_.value("log_scale", false);
        cd.contrast_min_slice_xy = meta_data_.value("contrast_min", 0.0f);
        cd.contrast_max_slice_xy = meta_data_.value("contrast_max", 0.0f);
        cd.fft_shift_enabled = meta_data_.value("fft_shift_enabled", true);
        cd.x_accu_enabled = meta_data_.value("x_acc_enabled", false);
        cd.x_acc_level = meta_data_.value("x_acc_level", 1);
        cd.y_accu_enabled = meta_data_.value("y_acc_enabled", false);
        cd.y_acc_level = meta_data_.value("y_acc_level", 1);
        cd.p_accu_enabled = meta_data_.value("p_acc_enabled", false);
        cd.p_acc_level = meta_data_.value("p_acc_level", 1);
        cd.img_acc_slice_xy_enabled = meta_data_.value("img_acc_slice_xy_enabled", false);
        cd.img_acc_slice_xz_enabled = meta_data_.value("img_acc_slice_xz_enabled", false);
        cd.img_acc_slice_yz_enabled = meta_data_.value("img_acc_slice_yz_enabled", false);
        cd.img_acc_slice_xy_level = meta_data_.value("img_acc_slice_xy_level", 1);
        cd.img_acc_slice_xz_level = meta_data_.value("img_acc_slice_xz_level", 1);
        cd.img_acc_slice_yz_level = meta_data_.value("img_acc_slice_yz_level", 1);
        cd.renorm_enabled = meta_data_.value("renorm_enabled", true);
        cd.renorm_constant = meta_data_.value("renorm_constant", 15);
    }
} // namespace holovibes::io_files