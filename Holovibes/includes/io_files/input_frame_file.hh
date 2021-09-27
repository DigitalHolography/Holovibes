/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_file.hh"

namespace holovibes
{
// Fast forward declaration
class ComputeDescriptor;

namespace io_files
{
/*! \class InputFrameFile
 *
 * \brief Base class of Input files
 */
class InputFrameFile : public FrameFile
{
  public:
    /*! \brief Update ComputeDescriptor with the settings present in the file
     *
     * \param cd The ComputeDescriptor to update
     */
    virtual void import_compute_settings(ComputeDescriptor& cd) const = 0;

    /*! \brief Set the pointer in the file to the frame requested
     *
     * This method must be called before the first read_frames call
     *
     * \throw FileException if an error occurred while setting the position
     */
    virtual void set_pos_to_frame(size_t frame_id) = 0;

    /*! \brief Read frames in the file
     *
     * \param buffer The allocated buffer in which the frames should be stored
     * \param frames_to_read The number of frames to read
     * \param flag_packed Return number of bit packed.
     * \return The actual number of frames read
     * \throw FileException if an error occurred while reading the file
     */
    virtual size_t read_frames(char* buffer, size_t frames_to_read, int* flag_packed);

  protected:
    // Give access to protected members to the handler
    friend class InputFrameFileFactory;

    /*! \brief Constructor
     *
     * Open the file in read only
     *
     * \param file_path The path of the file to open
     * \throw FileException if an error occurred while opening the file
     */
    InputFrameFile(const std::string& file_path);

    /*! \brief The size in bytes of a frame. Stored here to avoid computation at each call to read_frames */
    size_t frame_size_;

    /*! \brief The true size in bytes of a frame if image is packed (e.g. 10bit or 12bit ...) */
    size_t packed_frame_size_;
};
} // namespace io_files
} // namespace holovibes

#include "input_frame_file.hxx"
