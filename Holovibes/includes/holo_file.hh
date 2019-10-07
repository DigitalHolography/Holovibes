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
 * Holovibes file format
 *
 * -----------------------------------------------------------------------
 * | File header (sizeof(HoloFile::Header)) | Image data | File metadata |
 * -----------------------------------------------------------------------
 * | HOLO Magic number (4 bytes)            | Raw        | Metadata as   |
 * | Number of bits per pixel (2 bytes)     | image      | json format   |
 * | Image width (4 bytes)                  | data       |               |
 * | Image height (4 bytes)                 |            | #img / p / z  |
 * | Number of images (4 bytes)             |            | contrast / ...|
 * -----------------------------------------------------------------------
 *
 * Constant size header to open the files in ImageJ as raw with offset
 */

#pragma once

#include <string>
#include <cstdint>

#include "frame_desc.hh"
#include "compute_descriptor.hh"
#include "json.hh"
using json = ::nlohmann::json;

namespace holovibes
{
	/*! Used to get meta data from .holo files instead of file titles
	*
	* Reads the header of a file and if it is a .holo file reads the json
	* meta data stored at the end of the file and update the main window
	* ui with all the necessary data. This class behaves like a singleton */
	class HoloFile
	{
	public:
		/*! Packed 18 bytes .holo header to read the right amount bytes into it at once
		*
		* Only contains the necessarys information to retrieve the size of the binary images
		* data to skip directly to the meta data part at the end */
		#pragma pack(2)
		struct Header
		{
			/*! .holo file magic number, should be equal to "HOLO" */
			char HOLO[4];
			/*! Number of bits in 1 pixel */
			uint16_t pixel_bits;
			/*! Width of 1 image in pixels */
			uint32_t img_width;
			/*! Height of 1 image in pixels */
			uint32_t img_height;
			/*! Number of images in the file */
			uint32_t img_nb;
		};

		static HoloFile& new_instance(const std::string& file_path);
		static HoloFile& get_instance();

		/*! Returns the current file's header */
		const Header& get_header() const;
		/*! Returns the current file's meta data */
		const json& get_meta_data() const;
		/*! Sets the current file's meta data */
		void set_meta_data(const json& meta_data);

		/*! Returns true if the file is a .holo file */
		operator bool() const;

		/*! Creates a HoloFile::Header with the given arguments */
		static Header create_header(uint16_t pixel_bits, uint32_t img_width, uint32_t img_height, uint32_t img_nb = 0);

		/*! Updates a .holo file by replacing the meta data part
		*
		* \param meta_data_str Json meta data as a string */
		bool update(const std::string& meta_data_str);

		/*! Creates a .holo file
		*
		* \param header Header of the new .holo file, the img_nb field will be set according to the image and file sizes
		* \param meta_data_str Json meta data as a string
		* \param raw_file_path Path to the raw file to convert */
		static bool create(Header& header, const std::string& meta_data_str, const std::string& raw_file_path);

		/*! Returns a json object containing the settings from a compute descriptor
		*
		* \param cd Current compute descriptor */
		static json get_json_settings(const ComputeDescriptor& cd);

		/*! Returns a json object containing the settings from a frame descriptor and a compute descriptor
		*
		* \param fd Current frame descriptor
		* \param cd Current compute descriptor */
		static json get_json_settings(const ComputeDescriptor& cd, const camera::FrameDescriptor& fd);

	private:
		/*! Creates a HoloFile object from an existing file path and reads all of the required data
		*
		* \param file_path Path of the .holo file to process */
		HoloFile(const std::string& file_path);

		/*! Path of the .holo file */
		std::string holo_file_path_;

		/*! Header of the .holo file */
		Header header_;

		/*! Meta data offset in the file */
		uintmax_t meta_data_offset_;

		/*! True if header_.HOLO == "HOLO" */
		bool is_holo_file_ = false;

		/*! The json meta data as a std::string */
		std::string meta_data_str_;

		/*! The json meta data as a json object */
		json meta_data_;

		/*! Helper method to write data to a .holo file (used by create & update methods)
		*
		* \param header Header of the new .holo file, the img_nb field will be replaced
		* \param meta_data_str Json meta data as a string
		* \param data_file_path Path to the file containing image data (could be .raw or .holo)
		* \param output_path Path of the generated output file
		* \param begin_offset Offset to the beginning of the image data
		* \param end_offset Offset to the end of the image data */
		static bool write_holo_data(Header& header, const std::string& meta_data_str, const std::string& data_file_path, const std::string& output_path, fpos_t begin_offset, fpos_t end_offset);

		/*! Singleton instance */
		static HoloFile* instance;
	};
}