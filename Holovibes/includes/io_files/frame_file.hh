/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_desc.hh"

namespace holovibes::io_files
{
/*!
 *  \brief    Base class of Input and Output files
 *
 *  \details  Some methods, such as the constructors and the destructors,
 *            are protected to prevent the developers to access a file
 *            outside the file system.
 *            An UML diagram, called file_system, is provided in
 *            the documentation folder
 */
class FrameFile
{
  public:
    /*!
     *  \brief    Getter on the frame descriptor of the file
     */
    camera::FrameDescriptor get_frame_descriptor();

    /*!
     *  \brief    Getter on the frame descriptor of the file
     */
    const camera::FrameDescriptor& get_frame_descriptor() const;

    /*!
     *  \brief    Getter on the total number of frames in the file
     */
    virtual size_t get_total_nb_frames() const = 0;

  protected:
    /*!
     *  \brief    Enum representing the opening mode of a file
     */
    enum class OpeningMode
    {
        READ,
        WRITE
    };

    /*!
     *  \brief    Constructor
     *
     *  \details  Open the file with the desired mode
     *
     *  \param    file_path    The path of the file to open
     *  \param    mode         The opening mode of the file
     *
     *  \throw    FileException if an error occurred while opening the file
     */
    FrameFile(const std::string& file_path, OpeningMode mode);

    /*!
     *  \brief    Destructor
     *
     *  \details  Close the file
     */
    virtual ~FrameFile();

    /*!
     *  \brief    Default copy constructor
     */
    FrameFile(const FrameFile&) = default;

    /*!
     *  \brief    Default copy operator
     */
    FrameFile& operator=(const FrameFile&) = default;

    //! The path of the file
    const std::string file_path_;
    //! Frame descriptor associated to the file
    camera::FrameDescriptor fd_;
    //! Pointer associated to the file
    //! C way because it is faster
    std::FILE* file_ = nullptr;
};
} // namespace holovibes::io_files

#include "frame_file.hxx"
