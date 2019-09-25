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
	/*! \brief Used to get meta data from .holo files instead of file titles
	*
	* Reads the header of a file and if it is a .holo file reads the json
	* meta data stored at the end of the file and update the main window
	* ui with all the necessary data */
	class HoloFile
	{
	public:
		// nlohmann json lib
		using json = ::nlohmann::json;

		/*! \brief Packed .holo header to read the good amount bytes into it at once 
		*
		* Only contains the necessarys information to retrieve the size of the binary images
		* data to skip directly to the meta data part at the end */
		#pragma pack(2)
		struct Header
		{
			/*! \brief .holo file magic number, should be equal to "HOLO" */
			char HOLO[4];
			/*! \brief Number of bits in 1 pixel */
			uint16_t pixel_bits;
			/*! \brief Width of 1 image in pixels */
			uint32_t img_width;
			/*! \brief Height of 1 image in pixels */
			uint32_t img_height;
			/*! \brief Number of images in the file */
			uint32_t img_nb;
		};

		/*! \brief Creates a HoloFile object from an existing file path and reads all of the required data
		*
		* \param file_path Path of the .holo file to process */
		HoloFile(const std::string& file_path);

		/*! \brief Updates the MainWindow ui object with the .holo file data
		*
		* \param ui ui object contained in MainWindow */
		void update_ui(Ui::MainWindow& ui) const;

		/*! \brief Returns true if the file is a .holo file */
		operator bool() const;

		/*! Creates a .holo file
		*
		* \param header Header of the new .holo file, the img_nb field will be replaced
		* \param meta_data_str Json meta data as a string
		* \param raw_file_path Path to the raw file to convert */
		static bool create_holo_file(Header& header, const std::string& meta_data_str, const std::string& raw_file_path);

	private:
		/*! \brief Path of the .holo file */
		const std::string& holo_file_path;

		/*! \brief Header of the .holo file */
		Header header_;

		/*! \brief True if header_.HOLO == "HOLO" */
		bool is_holo_file_ = false;

		/*! \brief The json meta data as a char vector */
		std::vector<char> meta_data_str_;

		/*! The json meta data as a json object */
		json meta_data_;
	};
}