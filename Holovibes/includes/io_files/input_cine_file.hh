/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "input_frame_file.hh"
#include "cine_file.hh"

namespace holovibes::io_files
{
/*! \class InputCineFile
 *
 * \brief #TODO Add a description for this class
 */

/* FIXME: this class could and should be improved.
 * It handles the cine files like the previous versions of Holovibes,
 * but it does not handle all the specificities of the cine files (see link in
 * cine_file.hh).
 */
class InputCineFile : public InputFrameFile, public CineFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const override { return CineFile::get_total_nb_frames(); }

    /*! \brief Update ComputeDescriptor with the settings present in the file
     *
     *  \param cd The ComputeDescriptor to update
     */
    void import_compute_settings() const override;

    /*! \brief Set the pointer in the file to the frame requested
     *
     * This method is mandatory to read frames.
     *
     * \throw FileException if an error occurred while setting the position
     */
    void set_pos_to_frame(size_t frame_id) override;

    /*!
     *  \brief Read frames in the file
     *
     *  \param buffer The allocated buffer in which the frames should be stored
     *  \param frames_to_read The number of frames to read
     *  \param flag_packed Return number of bit packed.
     *  \return The actual number of frames read
     *  \throw FileException if an error occurred while reading the file
     */
    size_t read_frames(char* buffer, size_t frames_to_read, int* flag_packed) override;

  private:
    // Give access to private members to the factory
    friend class InputFrameFileFactory;

    /*! \brief Constructor
     *
     * Open the file and read all the required data
     *
     * \param file_path The path of the file to open
     * \throw FileException if an error occurred while opening or reading the file
     */
    InputCineFile(const std::string& file_path);
};
} // namespace holovibes::io_files
