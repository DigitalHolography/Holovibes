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

/*! \file
 *
 * Holovibes file format */

#pragma once

#include <string>
#include <cstdint>
#include <vector>

#include "json.hh"
#include "MainWindow.hh"

namespace holovibes
{
	class HoloFile
	{
	public:
		using json = ::nlohmann::json;

		#pragma pack(2)
		struct Header
		{
			char HOLO[4];
			uint16_t pixel_bits;
			uint32_t img_width;
			uint32_t img_height;
			uint32_t img_nb;
		};

		HoloFile(const std::string& file_path);
		const Header get_header() const;
		void update_ui(Ui::MainWindow& ui) const;

		operator bool() const;

	private:
		Header header_;
		bool is_holo_file_ = false;
		std::vector<char> meta_data_str_;
		json meta_data_;
	};
}