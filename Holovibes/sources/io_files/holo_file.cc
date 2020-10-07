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

#include "holo_file.hh"

#include <iostream>
#include <ios>
#include <fstream>
#include <filesystem>
#include <cstring>

#include "logger.hh"

namespace holovibes
{
	HoloFile* HoloFile::instance = nullptr;
	const uint16_t HoloFile::current_version = 2;

	HoloFile* HoloFile::new_instance(const std::string& file_path)
	{
		if (instance != nullptr)
			delete instance;

		instance = new HoloFile(file_path);

		if (!instance->is_valid_instance_)
			delete_instance();

		return instance;
	}

	HoloFile* HoloFile::get_instance()
	{
		if (instance == nullptr)
			LOG_WARN("HoloFile instance is null (get_instance)");

		return instance;
	}

	void HoloFile::delete_instance()
	{
		delete instance;
		instance = nullptr;
	}

	HoloFile::HoloFile(const std::string& file_path)
		: holo_file_path_(file_path)
	{
		std::ifstream file(file_path, std::ios::in | std::ios::binary);

		if (!file)
		{
			LOG_ERROR("Could not open file: " + file_path);
			return;
		}

		// read the file header
		file.read(reinterpret_cast<char*>(&header_), sizeof(Header));
		if (file.gcount() != sizeof(Header) || std::strncmp("HOLO", header_.HOLO, 4) != 0)
		{
			LOG_ERROR("Invalid HOLO file");
			return;
		}

		// compute the offset to retrieve the meta data
		meta_data_offset_ = sizeof(Header) + header_.total_data_size;
		uintmax_t file_size = std::filesystem::file_size(file_path);
		uintmax_t meta_data_size = file_size - meta_data_offset_;

		// retrieve the meta data
		if (meta_data_size > 0)
		{
			// handle crash if meta_data_size is greater than max_size()
			try
			{
				meta_data_str_.resize(meta_data_size + 1);
			}
			catch (std::length_error)
			{
				LOG_ERROR("An error was encountered while reading the file");
				return;
			}

			meta_data_str_[meta_data_size] = 0;

			file.seekg(meta_data_offset_, std::ios::beg);
			file.read(meta_data_str_.data(), meta_data_size);

			if (file.bad() || file.fail())
			{
				LOG_ERROR("An error was encountered while reading the file");
				return;
			}

			try
			{
				if (meta_data_size == 0)
					meta_data_str_ = "{}";

				meta_data_ = json::parse(meta_data_str_);
			}
			catch (const json::exception&)
			{
				LOG_WARN("Could not parse .holo file json meta data, settings are not imported");
			}
		}

		is_valid_instance_ = true;
		LOG_INFO("Loaded holo file: " + file_path + ", detected version: " + std::to_string(header_.version));
	}

	const HoloFile::Header& HoloFile::get_header() const
	{
		return header_;
	}

	const json& HoloFile::get_meta_data() const
	{
		return meta_data_;
	}

	HoloFile::Header HoloFile::create_header(uint16_t pixel_bits, uint32_t img_width, uint32_t img_height, uint32_t img_nb)
	{
		Header header;
		header.HOLO[0] = 'H';
		header.HOLO[1] = 'O';
		header.HOLO[2] = 'L';
		header.HOLO[3] = 'O';

		header.version = current_version;
		header.pixel_bits = pixel_bits;
		header.img_width = img_width;
		header.img_height = img_height;
		header.img_nb = img_nb;
		header.endianess = camera::Endianness::LittleEndian;

		header.total_data_size = (pixel_bits / 8);
		header.total_data_size *= img_width;
		header.total_data_size *= img_height;
		header.total_data_size *= img_nb;

		return header;
	}

	bool HoloFile::create(Header& header, const std::string& meta_data_str, const std::string& raw_file_path)
	{
		try
		{
			if (std::strncmp("HOLO", header.HOLO, 4) != 0)
			{
				LOG_WARN("header is not a .holo header");
				return false;
			}

			// Throws an exception if the file doesn't exist
			uintmax_t file_size = std::filesystem::file_size(raw_file_path);
			header.img_nb = file_size / (header.img_width * header.img_height * (header.pixel_bits / 8));
			header.total_data_size = file_size;

			// Throws an exception if the json string contains mistakes
			json meta_data = json::parse(meta_data_str);

			std::string output_path = raw_file_path.substr(0, raw_file_path.find_last_of('.')) + ".holo";

			LOG_INFO("Creating file: " + output_path);
			bool ret = write_holo_data(header, meta_data_str, raw_file_path, output_path, 0, file_size);
			LOG_INFO("Done.");
			return ret;
		}
		catch (const std::exception& e)
		{
			LOG_ERROR(e.what());
			return false;
		}
	}

	bool HoloFile::write_holo_data(Header& header, const std::string& meta_data_str, const std::string& data_file_path, const std::string& output_path, fpos_t begin_offset, fpos_t end_offset)
	{
		// Doing this the C way because it is much faster
		FILE* output;
		FILE* input;
		if (fopen_s(&output, output_path.c_str(), "wb") != 0)
		{
			LOG_WARN("Could not open output file: " + output_path);
			return false;
		}
		if (fopen_s(&input, data_file_path.c_str(), "rb") != 0)
		{
			LOG_WARN("Could not open input file: " + data_file_path);
			std::fclose(output);
			return false;
		}

#define UPDATE_BUF_SIZE 1 << 16
		std::fseek(input, begin_offset, SEEK_SET);
		std::fwrite(&header, sizeof(Header), 1, output);
		char buffer[UPDATE_BUF_SIZE];
		size_t data_size = end_offset - begin_offset;
		size_t r = 0;
		size_t w = 0;
		unsigned old_percent = 0;
		unsigned percent = 0;
		while (w < data_size)
		{
			// If the remaining data is less then UPDATE_BUF_SIZE only read what is necessary
			size_t to_read = data_size - r > UPDATE_BUF_SIZE ? UPDATE_BUF_SIZE : data_size - r;
			r = std::fread(buffer, 1, to_read, input);
			w += std::fwrite(buffer, 1, r, output);
			percent = w * 100 / data_size;
			if (percent - old_percent >= 10 || percent == 100)
			{
				LOG_INFO("Creating " + output_path + ": " + std::to_string(percent) + "%");
				old_percent = percent;
			}
		}
		std::fwrite(meta_data_str.data(), 1, meta_data_str.size(), output);
#undef UPDATE_BUF_SIZE

		std::fclose(output);
		std::fclose(input);

		return true;
	}

	json HoloFile::get_json_settings(const ComputeDescriptor& cd)
	{
		try
		{
			Computation mode = Computation::Direct;

			if (cd.record_raw.load() && cd.compute_mode.load() == Computation::Hologram)
				mode = Computation::Hologram;

			return json
			{
				{"mode", mode},

				{"algorithm", cd.algorithm.load()},
				{"time_filter", cd.time_filter.load()},

				{"#img", cd.nSize.load()},
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
		catch (const std::exception& e)
		{
			LOG_ERROR(e.what());
			return json();
		}
	}

	json HoloFile::get_json_settings(const ComputeDescriptor& cd, const camera::FrameDescriptor& fd)
	{
		try
		{
			json json_settings = HoloFile::get_json_settings(cd);
			json_settings.emplace("img_width", fd.width);
			json_settings.emplace("img_height", fd.height);
			json_settings.emplace("pixel_bits", fd.depth * 8);
			return json_settings;
		}
		catch (const std::exception& e)
		{
			LOG_ERROR(e.what());
			return json();
		}
	}
}
