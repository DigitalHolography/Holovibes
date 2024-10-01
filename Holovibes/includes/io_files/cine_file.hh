/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

namespace holovibes::io_files
{
/*! \class CineFile
 *
 * \brief Base class of cine files. Used to store data
 *
 * \details For more details, see:
 * http://phantomhighspeed-knowledge.force.com/servlet/fileField?id=0BE1N000000kD2i#:~:text=Cine%20file%20format%20was%20designed,cameras%20from%20Vision%20Research%20Inc.
 *
 * More structs present in the file format could be added to the class to perform more checks
 */
// FIXME: This class may be completed by adding more structures
class CineFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const { return cine_file_header_.image_count; }

  protected:
/*! \struct CineFileHeader
 *
 * \brief Struct containing data related directly to the cine file
 *
 * Packed (aligned on 2 bytes) to be exactly 44 bytes, as in the file
 */
#pragma pack(2)
    struct CineFileHeader
    {
        /*! \brief Marker of a cine file. It has to be "CI" in any cine file */
        uint16_t type;
        /*! \brief It represents the CINEFILEHEADER structure size as a number of bytes */
        uint16_t header_size;
        /*! \brief 0 for gray cines, 1 for a jpeg compressed file, 2 for uninterpolated (RAW file) */
        uint16_t compression;
        /*! \brief Version number */
        uint16_t version;
        /*! \brief First recorded image number, relative to trigger */
        int32_t first_movie_image;
        /*! \brief Total count of images, recorded in the camera memory */
        uint32_t total_image_count;
        /*! \brief First image saved to this file, relative to trigger */
        int32_t first_image_no;
        /*! \brief Count of images saved to this file */
        uint32_t image_count;
        /*! \brief Offset of the BITMAPINFOHEADER structure in the cine file */
        uint32_t off_image_header;
        /*! \brief Offset of the SETUP structure in the cine file */
        uint32_t off_setup;
        /*! \brief Offset in the cine file of an array with the positions of each image stored in the file */
        uint32_t off_image_offset;
        /*! \brief Trigger time is a TIME64 structure having the seconds and fraction of second since Jan 1, 1970 */
        time_t trigger_time;
    };

/*! \struct BitmapInfoHeader
 *
 * \brief Struct containing data related directly to the frames
 *
 * Packed (aligned on 2 bytes) to be exactly 40 bytes, as in the file
 */
#pragma pack(2)
    struct BitmapInfoHeader
    {
        /*! \brief It specifies the number of bytes required by the structure (without palette) */
        uint32_t bi_size;
        /*! \brief It specifies the width of the bitmap, in pixels */
        int32_t bi_width;
        /*! \brief It specifies the height of the bitmap, in pixels
         *
         * If biHeight is positive, the bitmap is a bottom-up DIB and its origin is the lower-left corner.
         * If biHeight is negative, the bitmap is a top-down DIB and its origin is the upper-left corner
         */
        int32_t bi_height;
        /*! \brief It specifies the number of planes for the target device. This value must be set to 1 */
        uint16_t bi_planes;
        /*! \brief It specifies the number of bits-per-pixel */
        uint16_t bi_bit_count;
        /*! \brief It specifies the type of compression for a compressed bottom-up bitmap */
        uint32_t bi_compression;
        /*! \brief It specifies the image size in bytes */
        uint32_t bi_size_image;
        /*! \brief It specifies the horizontal resolution, in pixels-per-meter, of the target device for the bitmap */
        int32_t bi_x_pels_per_meter;
        /*! \brief Vertical resolution in pixels per meter */
        int32_t bi_y_pels_per_meter;
        /*! \brief Specifies the number of color indexes in the color table that are actually used by the bitmap */
        uint32_t bi_clr_used;
        /*! \brief It specifies the number of color indexes that are required for displaying the bitmap */
        uint32_t bi_clr_important;
    };

    /*! \brief Default constructor */
    CineFile() = default;

    /*! \brief Abstract destructor to make class abstract */
    virtual ~CineFile() {};

    /*! \brief Default copy constructor */
    CineFile(const CineFile&) = default;

    /*! \brief Default copy operator */
    CineFile& operator=(const CineFile&) = default;

    /*! \brief The cine file header of the cine file */
    CineFileHeader cine_file_header_;
    /*! \brief The bitmap info header of the cine file */
    BitmapInfoHeader bitmap_info_header_;
};
} // namespace holovibes::io_files
