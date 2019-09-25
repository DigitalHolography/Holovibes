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

#include "logger.hh"

namespace holovibes
{
	HoloFile::HoloFile(const std::string& file_path)
	{
		std::ifstream file(file_path, std::ios::in | std::ios::binary);

		if (!file)
		{
			LOG_ERROR("Could not open file: " + file_path);
			return;
		}

		file.read((char*)(&header_), sizeof(Header));
		if (file.gcount() != sizeof(Header))
		{
			return;
		}

		is_holo_file_ = std::strncmp("HOLO", header_.HOLO, 4) == 0;
		if (!is_holo_file_)
		{
			return;
		}

		uintmax_t meta_data_offset = sizeof(Header) + (header_.img_height * header_.img_width * header_.img_nb * (header_.pixel_bits / 8));
		uintmax_t file_size = std::filesystem::file_size(file_path);
		uintmax_t meta_data_size = file_size - meta_data_offset;

		meta_data_str_.resize(meta_data_size + 1);
		meta_data_str_[meta_data_size] = 0;

		file.seekg(meta_data_offset, std::ios::beg);
		file.read(meta_data_str_.data(), meta_data_size);
		meta_data_ = json::parse(meta_data_str_);
	}

	void HoloFile::update_ui(Ui::MainWindow& ui) const
	{
		if (!is_holo_file_)
			return;

		QSpinBox *import_width_box = ui.ImportWidthSpinBox;
		QSpinBox *import_height_box = ui.ImportHeightSpinBox;
		QComboBox *import_depth_box = ui.ImportDepthComboBox;
		QComboBox *import_endian_box = ui.ImportEndiannessComboBox;

		import_width_box->setValue(header_.img_width);
		import_height_box->setValue(header_.img_height);
		import_depth_box->setCurrentIndex(log2(header_.pixel_bits) - 3);
		import_endian_box->setCurrentIndex(0);
	}

	HoloFile::operator bool() const
	{
		return is_holo_file_;
	}
}