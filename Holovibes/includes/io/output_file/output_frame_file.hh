/*! \file output_frame_file.hh
 *
 * \brief Defines the OutputFrameFile class, the base class for all output files class.
 */
#pragma once

#include "frame_file.hh"
#include "logger.hh"

namespace holovibes::io_files
{
/*! \class OutputFrameFile
 *
 *  \brief Base class of Output files
 */
class OutputFrameFile : public FrameFile
{
  public:
    /*! \brief Export the compute settings in the file
     *
     * \param input_fps how fast the input was
     * \param contiguous How many frames were contiguous in input
     */
    virtual void export_compute_settings(int input_fps, size_t contiguous, int camera_fps) = 0;

    /*! \brief Write the header in the file
     *
     * \throw FileException if an error occurred while writing the header
     */
    virtual void write_header() = 0;

    // FIXME: update write_frame method, try to remove frame_size
    /*! \brief Write a frame in the file
     *
     * \param frame The allocated buffer containing the frame
     * \param frame_size The size in bytes of a frame
     * \return The number of bytes written in the file
     * \throw FileException if an error occurred while writing the frame
     */
    virtual size_t write_frame(const char* frame, size_t frame_size) = 0;

    /*! \brief Write the footer in the file
     *
     * \throw FileException if an error occurred while writing the footer
     */
    virtual void write_footer() = 0;

    /*! \brief Rewrite the sections in the file where the number of frames has been used
     *
     * For example, it is useful to correct the header when the file is written with a
     * different number of frames than the expected number of frames
     *
     * \throw FileException if an error occurred while correcting the sections
     */
    virtual void correct_number_of_frames(size_t nb_frames_written) = 0;

  protected:
    // Give access to protected members to the factory
    friend class OutputFrameFileFactory;

    /*! \brief Constructor
     *
     * Open the file in write only
     *
     * \param file_path The path of the file to open
     * \throw FileException if an error occurred while opening the file
     */
    OutputFrameFile(const std::string& file_path)
        : FrameFile(file_path, FrameFile::OpeningMode::WRITE)
    {
    }

    /*!
     * \brief Compute the output fps as follow: input_fps / time_stride.
     *
     * \return double return the compute output fps
     */
    double compute_output_fps();
};
} // namespace holovibes::io_files
