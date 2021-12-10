/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "input_frame_file.hh"
#include "holo_file.hh"

namespace holovibes::io_files
{
/*! \class InputHoloFile
 *
 * \brief #TODO Add a description for this class
 */
class InputHoloFile : public InputFrameFile, public HoloFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const override { return HoloFile::get_total_nb_frames(); }

    /*! \brief Set the pointer in the file to the frame requested
     *
     * This method is mandatory to read frames.
     *
     * \throw FileException if an error occurred while setting the position
     */
    void set_pos_to_frame(size_t frame_id) override;

    /*! \brief Update ComputeDescriptor with the settings present in the file
     *
     * \param cd The ComputeDescriptor to update
     */
    void import_compute_settings(ComputeDescriptor& cd) const override;

    /*! \brief Update ComputeDescriptor with the settings present in the file
     *
     * \param cd The ComputeDescriptor to update
     */
    void import_info(ComputeDescriptor& cd) const override;

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
    InputHoloFile(const std::string& file_path);
};
} // namespace holovibes::io_files
