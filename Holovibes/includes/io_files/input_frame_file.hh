/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "frame_file.hh"

namespace holovibes
{
// Fast forward declaration
class ComputeDescriptor;

namespace io_files
{
/*!
 *  \brief    Base class of Input files
 */
class InputFrameFile : public FrameFile
{
  public:
    /*!
     *  \brief    Update ComputeDescriptor with the settings present in the file
     *
     *  \param    cd    The ComputeDescriptor to update
     */
    virtual void import_compute_settings(ComputeDescriptor& cd) const = 0;

    /*!
     *  \brief    Set the pointer in the file to the frame requested
     *  \details  This method must be called before the first read_frames call
     *
     *  \throw    FileException if an error occurred while setting the position
     */
    virtual void set_pos_to_frame(size_t frame_id) = 0;

    /*!
     *  \brief    Read frames in the file
     *
     *  \param    buffer            The allocated buffer in which the frames
     * should be stored \param    frames_to_read    The number of frames to read
     *
     *  \return   The actual number of frames read
     *
     *  \throw    FileException if an error occurred while reading the file
     */
    virtual size_t read_frames(char* buffer, size_t frames_to_read);

  protected:
    // Give access to protected members to the handler
    friend class InputFrameFileFactory;

    /*!
     *  \brief    Constructor
     *
     *  \details  Open the file in read only
     *
     *  \param    file_path    The path of the file to open
     *
     *  \throw    FileException if an error occurred while opening the file
     */
    InputFrameFile(const std::string& file_path);

    //! The size in bytes of a frame
    //! Stored here to avoid computation at each call to read_frames
    size_t frame_size_;
};
} // namespace io_files
} // namespace holovibes

#include "input_frame_file.hxx"
