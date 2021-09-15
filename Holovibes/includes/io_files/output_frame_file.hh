/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_file.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
/*! \class OutputFrameFile
 *
 *  \brief Base class of Output files
 */
class OutputFrameFile : public FrameFile
{
  public:
    /*!
     *  \brief    Export the compute settings in the file
     *
     *  \param    cd            The ComputeDescriptor containing the compute
     * settings \param    record_raw    Is the raw record enabled
     */
    virtual void export_compute_settings(const ComputeDescriptor& cd,
                                         bool record_raw) = 0;

    /*!
     *  \brief    Write the header in the file
     *
     *  \throw    FileException if an error occurred while writing the header
     */
    virtual void write_header() = 0;

    // FIXME: update write_frame method, try to remove frame_size
    /*!
     *  \brief    Write a frame in the file
     *
     *  \param    frame        The allocated buffer containing the frame
     *  \param    frame_size   The size in bytes of a frame
     *
     *  \return   The number of bytes written in the file
     *
     *  \throw    FileException if an error occurred while writing the frame
     */
    virtual size_t write_frame(const char* frame, size_t frame_size) = 0;

    /*!
     *  \brief    Write the footer in the file
     *
     *  \throw    FileException if an error occurred while writing the footer
     */
    virtual void write_footer() = 0;

    /*!
     *  \brief    Rewrite the sections in the file where the number of frames
     * has been used \details  For example, it is useful to correct the header
     * when the file is written with a different number of frames than the
     * expected number of frames
     *
     *  \throw    FileException if an error occurred while correcting the
     * sections
     */
    virtual void correct_number_of_frames(size_t nb_frames_written) = 0;

    virtual void set_make_square_output(bool make_square_output);

  protected:
    // Give access to protected members to the factory
    friend class OutputFrameFileFactory;

    /*!
     *  \brief    Constructor
     *
     *  \details  Open the file in write only
     *
     *  \param    file_path    The path of the file to open
     *
     *  \throw    FileException if an error occurred while opening the file
     */
    OutputFrameFile(const std::string& file_path);

    std::optional<unsigned short> max_side_square_output_ = std::nullopt;
};
} // namespace holovibes::io_files

#include "output_frame_file.hxx"
