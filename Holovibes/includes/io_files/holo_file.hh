/*! \file
 *
 * \brief #TODO Add a description for this file
 *
 * Holovibes file format
 *
 * \verbatim
 * -----------------------------------------------------------------------
 * | File header (sizeof(HoloFile::Header)) | Image data | File metadata |
 * -----------------------------------------------------------------------
 * | HOLO Magic number (4 bytes)            | Raw        | Metadata as   |
 * | Version number (2 bytes)               | image      |               |
 * | Number of bits per pixel (2 bytes)     | data       | json format   |
 * | Image width (4 bytes)                  |            |               |
 * | Image height (4 bytes)                 |            | #img / p / z  |
 * | Number of images (4 bytes)             |            | contrast / ...|
 * | Total data size (8 bytes)              |            |               |
 * | Padding up to 64 bytes                 |            |               |
 * -----------------------------------------------------------------------
 * \endverbatim
 *
 * Constant size header to open the files in ImageJ as raw with offset
 */

#pragma once

#include <sstream>

#include <nlohmann/json.hpp>
#include "compute_settings_struct.hh"
using json = ::nlohmann::json;

/*! \brief #TODO Add a description for this namespace */
namespace holovibes::io_files
{
/*! \class HoloFile
 *
 * \brief Base class of holo files. Used to store data
 */
class HoloFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const { return holo_file_header_.img_nb; }

  protected:
    /*! \brief Struct containing data related directly to the holo file
     *
     * Packed (aligned on 2 bytes) to be exactly 64 bytes
     */
#pragma pack(2)
    struct HoloFileHeader
    {
        /*! \brief .holo file magic number, should be equal to "HOLO" */
        char magic_number[4];
        /*! \brief Version number, starts at 0 */
        uint16_t version;
        /*! \brief Number of bits in 1 pixel */
        uint16_t bits_per_pixel;
        /*! \brief Width of 1 image in pixels */
        uint32_t img_width;
        /*! \brief Height of 1 image in pixels */
        uint32_t img_height;
        /*! \brief Number of images in the file */
        uint32_t img_nb;
        /*! \brief Total size of the data in bytes img_width * img_height * nb_img * (bits_per_pixel / 8) */
        uint64_t total_data_size;
        /*! \brief Data endianness */
        uint8_t endianness;
        /*! \brief Padding to make the header 64 bytes long */
        char padding[35];
    };

    /*! \brief Default constructor */
    HoloFile() = default;

    /*! \brief Abstract destructor to make class abstract */
    virtual ~HoloFile(){};

    /*! \brief Default copy constructor */
    HoloFile(const HoloFile&) = default;

    /*! \brief Default copy operator */
    HoloFile& operator=(const HoloFile&) = default;

    /*! \brief Header of the holo file */
    HoloFileHeader holo_file_header_;
    /*! \brief The json meta data present in the footer */
    json meta_data_;
    /*! \brief The json meta data present in the footer */
    ComputeSettings raw_footer_;
    /*! \brief Current version of the holo file, update it when changing version */
    static constexpr uint16_t current_version_ = 5;
};
} // namespace holovibes::io_files
