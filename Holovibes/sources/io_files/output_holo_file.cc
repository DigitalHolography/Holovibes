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

#include "output_holo_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
    OutputHoloFile::OutputHoloFile(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb):
        OutputFile(file_path),
        HoloFile()
    {
        holo_file_header_.magic_number[0] = 'H';
		holo_file_header_.magic_number[1] = 'O';
		holo_file_header_.magic_number[2] = 'L';
		holo_file_header_.magic_number[3] = 'O';

		holo_file_header_.version = current_version_;
		holo_file_header_.bits_per_pixel = fd.depth * 8;
		holo_file_header_.img_width = fd.width;
		holo_file_header_.img_height = fd.height;
		holo_file_header_.img_nb = img_nb;
		holo_file_header_.endianness = camera::Endianness::LittleEndian;

		holo_file_header_.total_data_size = fd.frame_size() * img_nb;

        meta_data_ = json();
    }

    void OutputHoloFile::export_compute_settings(const ComputeDescriptor& cd)
    {
		// export as a json
		try
		{
			Computation mode = Computation::Raw;

			if (cd.record_raw.load() && cd.compute_mode.load() == Computation::Hologram)
				mode = Computation::Hologram;

			meta_data_ = json
			{
				{"mode", mode},

				{"algorithm", cd.algorithm.load()},
				{"time_transformation", cd.time_transformation.load()},

				{"#img", cd.time_transformation_size.load()},
				{"p", cd.pindex.load()},
				{"lambda", cd.lambda.load()},
				{"pixel_size", cd.pixel_size.load()},
				{"z", cd.zdistance.load()},

				{"fft_shift_enabled", cd.fft_shift_enabled.load()},

				{"x_acc_enabled", cd.x_accu_enabled.load()},
				{"x_acc_level" , cd.x_acc_level.load()},
				{"y_acc_enabled", cd.y_accu_enabled.load()},
				{"y_acc_level" , cd.y_acc_level.load()},
				{"p_acc_enabled", cd.p_accu_enabled.load()},
				{"p_acc_level" , cd.p_acc_level.load()},

				{"log_scale", cd.log_scale_slice_xy_enabled.load()},
				{"contrast_min", cd.contrast_min_slice_xy.load()},
				{"contrast_max", cd.contrast_max_slice_xy.load()},

				{"img_acc_slice_xy_enabled", cd.img_acc_slice_xy_enabled.load()},
				{"img_acc_slice_xz_enabled", cd.img_acc_slice_xz_enabled.load()},
				{"img_acc_slice_yz_enabled", cd.img_acc_slice_yz_enabled.load()},
				{"img_acc_slice_xy_level", cd.img_acc_slice_xy_level.load()},
				{"img_acc_slice_xz_level", cd.img_acc_slice_xz_level.load()},
				{"img_acc_slice_yz_level", cd.img_acc_slice_yz_level.load()},

				{"renorm_enabled", cd.renorm_enabled.load()},
				{"renorm_constant", cd.renorm_constant.load()}
			};
		}
		catch (const json::exception& e)
		{
            meta_data_ = json();
            throw FileException("An error was encountered while trying to export compute settings");
		}
    }

    void OutputHoloFile::write_header()
    {
        if (std::fwrite(&holo_file_header_, sizeof(char), sizeof(HoloFileHeader), file_) != sizeof(HoloFileHeader))
            throw FileException("Unable to write output holo file header");
    }

    size_t OutputHoloFile::write_frame(const char* frame, size_t frame_size)
    {
        size_t written_bytes = std::fwrite(frame, sizeof(char), frame_size, file_);

        if (written_bytes != frame_size)
            throw FileException("Unable to write output holo file frame");

        return written_bytes;
    }

    void OutputHoloFile::write_footer()
    {
        const std::string& meta_data_str = meta_data_.dump();
        const size_t meta_data_size = meta_data_str.size();

        if (std::fwrite(meta_data_str.data(), sizeof(char), meta_data_size, file_ ) != meta_data_size)
            throw FileException("Unable to write output holo file footer");
    }
} // namespace holovibes::io_files