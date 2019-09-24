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

#include "logger.hh"

namespace holovibes
{
	HoloFile::HoloFile(const std::string& file_path)
	{
		fopen_s(&file_, file_path.c_str(), "rb");

		if (!file_)
		{
			LOG_ERROR("Could not open file: " + file_path);
			return;
		}

		if (std::fread(&header_, 1, sizeof(Header), file_) != sizeof(Header))
		{
			LOG_ERROR("Could not read header of file: " + file_path);
			return;
		}

		// printf("HOLO[4]: %.4s\n", header_.HOLO);
		// std::cout << "pixel_bits: " << header_.pixel_bits << "\n";
		// std::cout << "width: " << header_.img_width << "\n";
		// std::cout << "height: " << header_.img_height << "\n";
		// std::cout << "number of images: " << header_.img_nb << "\n";
	}

	HoloFile::~HoloFile()
	{
		if (file_)
		{
			std::fclose(file_);
		}
	}
}