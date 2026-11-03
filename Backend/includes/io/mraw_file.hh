#pragma once

#include <sstream>

namespace holovibes::io_files
{
/*! \class MrawFile
 *
 * \brief Base class of mraw files. Used to store data
 */
class MrawFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const { return img_nb; }

  protected:
    uint16_t bits_per_pixel;
    /*! \brief Width of 1 image in pixels */
    uint32_t img_width;
    /*! \brief Height of 1 image in pixels */
    uint32_t img_height;
    /*! \brief Number of images in the file */
    uint32_t img_nb;
    /*! \brief Total size of the data in bytes img_width * img_height * nb_img * (bits_per_pixel / 8) */
    uint64_t total_data_size;

    /*! \brief Default constructor */
    MrawFile() = default;

    /*! \brief Abstract destructor to make class abstract */
    virtual ~MrawFile(){};

    /*! \brief Default copy constructor */
    MrawFile(const MrawFile&) = default;

    /*! \brief Default copy operator */
    MrawFile& operator=(const MrawFile&) = default;
};
} // namespace holovibes::io_files
