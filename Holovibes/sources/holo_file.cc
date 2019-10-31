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
	const uint16_t HoloFile::current_version = 0;

	HoloFile& HoloFile::new_instance(const std::string& file_path)
	{
		if (instance != nullptr)
			delete instance;
		instance = new HoloFile(file_path);
		return *instance;
	}

	HoloFile& HoloFile::get_instance()
	{
		if (instance == nullptr)
			return new_instance("");
		return *instance;
	}

	HoloFile::HoloFile(const std::string& file_path)
		: holo_file_path_(file_path)
	{
		if (file_path == "")
			return;

		std::ifstream file(file_path, std::ios::in | std::ios::binary);

		if (!file)
		{
			LOG_ERROR("Could not open file: " + file_path);
			return;
		}

		file.read(reinterpret_cast<char*>(&header_), sizeof(Header));
		if (file.gcount() != sizeof(Header))
		{
			return;
		}

		is_holo_file_ = std::strncmp("HOLO", header_.HOLO, 4) == 0;
		if (!is_holo_file_)
		{
			return;
		}

		meta_data_offset_ = sizeof(Header) + header_.total_data_size;
		uintmax_t file_size = std::filesystem::file_size(file_path);
		uintmax_t meta_data_size = file_size - meta_data_offset_;

		meta_data_str_.resize(meta_data_size + 1);
		meta_data_str_[meta_data_size] = 0;

		file.seekg(meta_data_offset_, std::ios::beg);
		file.read(meta_data_str_.data(), meta_data_size);

		try
		{
			meta_data_ = json::parse(meta_data_str_);
			LOG_INFO("Loaded holo file: " + file_path + ", detected version: " + std::to_string(header_.version));
		}
		catch (const json::exception&)
		{
			LOG_WARN("Could not parse .holo file json meta data, treating the file as a regular .raw file");
			is_holo_file_ = false;
			return;
		}
	}

	const HoloFile::Header& HoloFile::get_header() const
	{
		return header_;
	}

	const json& HoloFile::get_meta_data() const
	{
		return meta_data_;
	}

	void HoloFile::set_meta_data(const json& meta_data)
	{
		meta_data_ = meta_data;
	}

	HoloFile::operator bool() const
	{
		return is_holo_file_;
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

		header.total_data_size = (pixel_bits / 8);
		header.total_data_size *= img_width;
		header.total_data_size *= img_height;
		header.total_data_size *= img_nb;

		return header;
	}


	bool HoloFile::update(const std::string& meta_data_str)
	{
		try
		{
			if (!is_holo_file_)
			{
				LOG_WARN(holo_file_path_ + " is not a .holo file, it cannot be updated");
				return false;
			}

			json meta_data = json::parse(meta_data_str);

			LOG_INFO("Copying and updating file: " + holo_file_path_);

			size_t last_dir = holo_file_path_.find_last_of('/');
			std::string new_file_path;
			if (last_dir == holo_file_path_.npos)
			{
				new_file_path = "copy_" + holo_file_path_;
			}
			else
			{
				std::string dir = holo_file_path_.substr(0, last_dir + 1);
				std::string file = holo_file_path_.substr(last_dir + 1, holo_file_path_.size() - last_dir);
				new_file_path = dir + "copy_" + file;
			}

			bool ret = write_holo_data(header_, meta_data_str, holo_file_path_, new_file_path, sizeof(Header), meta_data_offset_);

			if (ret)
			{
				LOG_INFO("Done.");
			}

			return true;
		}
		catch (const std::exception& e)
		{
			LOG_ERROR(e.what());
			return false;
		}
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
				std::cout << "Creating " << output_path << ": " << percent << "%" << std::endl;
				old_percent = percent;
			}
		}
		std::fwrite(meta_data_str.data(), 1, meta_data_str.size(), output);
#undef BUF_SIZE

		std::fclose(output);
		std::fclose(input);

		return true;
	}

	json HoloFile::get_json_settings(const ComputeDescriptor& cd)
	{
		try
		{
			return json
			{
				{"algorithm", cd.algorithm.load()},
				{"#img", cd.nSize.load()},
				{"p", cd.pindex.load()},
				{"lambda", cd.lambda.load()},
				{"pixel_size", cd.pixel_size.load()},
				{"z", cd.zdistance.load()},
				{"log_scale", cd.log_scale_slice_xy_enabled.load()},
				{"contrast_min", cd.contrast_min_slice_xy.load()},
				{"contrast_max", cd.contrast_max_slice_xy.load()}
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
			json_settings.emplace("endianess", fd.byteEndian);
			return json_settings;
		}
		catch (const std::exception& e)
		{
			LOG_ERROR(e.what());
			return json();
		}
	}
}