/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*!
 *
 * Holovibes file format
 *
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
 *
 * Constant size header to open the files in ImageJ as raw with offset
 */

#pragma once

#include <sstream>

#include "json.hh"

using json = ::nlohmann::json;

namespace holovibes::io_files
{
/*!
 *  \brief    Base class of holo files. Used to store data
 */
class HoloFile
{
  public:
    /*!
     *  \brief    Getter on the total number of frames in the file
     */
    size_t get_total_nb_frames() const;

  protected:
    /*!
     *  \brief    Struct containing data related directly to the holo file
     *
     *  \details  Packed (aligned on 2 bytes) to be exactly 64 bytes
     */
#pragma pack(2)
    struct HoloFileHeader
    {
        /*! .holo file magic number, should be equal to "HOLO" */
        char magic_number[4];
        /*! Version number, starts at 0 */
        uint16_t version;
        /*! Number of bits in 1 pixel */
        uint16_t bits_per_pixel;
        /*! Width of 1 image in pixels */
        uint32_t img_width;
        /*! Height of 1 image in pixels */
        uint32_t img_height;
        /*! Number of images in the file */
        uint32_t img_nb;
        /*! Total size of the data in bytes
         *  img_width * img_height * nb_img * (bits_per_pixel / 8) */
        uint64_t total_data_size;
        /*! Data endianness */
        uint8_t endianness;
        /*! Padding to make the header 64 bytes long */
        char padding[35];
    };

    /*!
     *  \brief    Default constructor
     */
    HoloFile() = default;

    /*!
     *  \brief    Abstract destructor to make class abstract
     */
    virtual ~HoloFile() = 0;

    /*!
     *  \brief    Default copy constructor
     */
    HoloFile(const HoloFile&) = default;

    /*!
     *  \brief    Default copy operator
     */
    HoloFile& operator=(const HoloFile&) = default;

    //! Header of the holo file
    HoloFileHeader holo_file_header_;
    //! The json meta data present in the footer */
    json meta_data_;
    //! Current version of the holo file, update it when changing version */
    static constexpr uint16_t current_version_ = 2;
};
} // namespace holovibes::io_files

#include "holo_file.hxx"
