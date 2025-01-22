/*! \file
 *
 * \brief Definition of the FrameFile class, the base class of other IO classes used by the Holovibes application.
 */
#pragma once

#include <cstdio>
#include <string>

#include "frame_desc.hh"

namespace holovibes::io_files
{
/*! \class FrameFile
 *
 * \brief Base class of Input and Output files
 *
 * Some methods, such as the constructors and the destructors,
 * are protected to prevent the developers to access a file outside the file system.
 * An UML diagram, called file_system, is provided in the documentation folder
 */
class FrameFile
{
  public:
    /*! \brief Getter on the frame descriptor of the file */
    camera::FrameDescriptor get_frame_descriptor() { return fd_; }

    /*! \brief Getter on the frame descriptor of the file */
    const camera::FrameDescriptor& get_frame_descriptor() const { return fd_; }

    /*! \brief Getter on the total number of frames in the file */
    virtual size_t get_total_nb_frames() const = 0;

    std::string get_file_path() { return file_path_; };

  protected:
    /*! \brief Enum representing the opening mode of a file */
    enum class OpeningMode
    {
        READ,
        WRITE
    };

    /*! \brief Open the file with the desired mode
     *
     * \param file_path The path of the file to open
     * \param mode The opening mode of the file
     *
     * \throw FileException if an error occurred while opening the file
     */
    FrameFile(const std::string& file_path, OpeningMode mode);

    /*! \brief Close the file */
    virtual ~FrameFile();

    /*! \brief The path of the file */
    const std::string file_path_;
    /*! \brief Frame descriptor associated to the file */
    camera::FrameDescriptor fd_;
    /*! \brief Pointer associated to the file. C way because it is faster */
    std::FILE* file_ = nullptr;
};
} // namespace holovibes::io_files
