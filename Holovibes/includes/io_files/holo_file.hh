/*! \file
 *
 * \brief Holovibes file format definition
 *
 * This file defines the structure and classes required to handle Holovibes (.holo) file format,
 * which is used to store image data and metadata.
 *
 * \verbatim
 * -----------------------------------------------------------------------
 * | File header (sizeof(HoloFile::Header)) | Image data | File metadata |
 * -----------------------------------------------------------------------
 * | HOLO Magic number (4 bytes)            | Raw        | Metadata as   |
 * | Version number (2 bytes)               | image      | JSON format   |
 * | Number of bits per pixel (2 bytes)     | data       |               |
 * | Image width (4 bytes)                  |            |               |
 * | Image height (4 bytes)                 |            | #img / p / z  |
 * | Number of images (4 bytes)             |            | contrast / ...|
 * | Total data size (8 bytes)              |            |               |
 * | Padding up to 64 bytes                 |            |               |
 * -----------------------------------------------------------------------
 * \endverbatim
 *
 * The header has a constant size, allowing the files to be opened in ImageJ as raw images with an offset.
 */

#pragma once

#include <sstream>
#include <nlohmann/json.hpp>
#include "compute_settings_struct.hh"
using json = ::nlohmann::json;

/*! \brief Namespace for Holovibes input/output file handling */
namespace holovibes::io_files
{
/*! \class HoloFile
 *
 * \brief Base class for .holo files, used to store image data and metadata.
 */
class HoloFile
{
  public:
    /*! \brief Gets the total number of frames (images) in the file */
    size_t get_total_nb_frames() const { return holo_file_header_.img_nb; }

  protected:
    /*! \brief Struct containing the header data of a .holo file
     *
     * This struct is packed to ensure it is exactly 64 bytes.
     */
#pragma pack(2)
    struct HoloFileHeader
    {
        /*! \brief .holo file magic number, should be equal to "HOLO" */
        char magic_number[4];
        /*! \brief Version number, starts at 0 */
        uint16_t version;
        /*! \brief Number of bits per pixel */
        uint16_t bits_per_pixel;
        /*! \brief Width of an image in pixels */
        uint32_t img_width;
        /*! \brief Height of an image in pixels */
        uint32_t img_height;
        /*! \brief Number of images in the file */
        uint32_t img_nb;
        /*! \brief Total size of the image data in bytes: img_width * img_height * img_nb * (bits_per_pixel / 8) */
        uint64_t total_data_size;
        /*! \brief Data endianness indicator */
        uint8_t endianness;
        /*! \brief Padding to make the header 64 bytes long */
        char padding[35];
    };

    /*! \brief Default constructor */
    HoloFile() = default;

    /*! \brief Abstract destructor to make the class abstract */
    virtual ~HoloFile() = default;

    /*! \brief Default copy constructor */
    HoloFile(const HoloFile&) = default;

    /*! \brief Default copy assignment operator */
    HoloFile& operator=(const HoloFile&) = default;

    /*! \brief Header of the .holo file */
    HoloFileHeader holo_file_header_;
    /*! \brief JSON metadata present in the file footer */
    json meta_data_;
    /*! \brief Compute settings present in the file footer */
    ComputeSettings raw_footer_;
    /*! \brief Current version of the .holo file format */
    static constexpr uint16_t current_version_ = 5;
};
} // namespace holovibes::io_files
