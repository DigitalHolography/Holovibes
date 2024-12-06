/*! \file
 *
 * \brief Handle loading of .holo files
 */
#pragma once

#include "input_frame_file.hh"
#include "holo_file.hh"

namespace holovibes::io_files
{
/*! \class InputHoloFile
 *
 * \brief Class that represents an holo file
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

    /*! \brief Update Global State Holder with the settings present in the file */
    void import_compute_settings() override;

    /*! \brief Update Global State Holder with the settings present in the file */
    void import_info() const override;

    /*! \brief Load json patch and apply it to meta_data */
    void apply_json_patch(json& meta_data, const std::string& json_patch_path);

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

    /*! \brief Open file to initialize holo_file_header_
     */
    void load_header();

    /*! \brief Open file to initialize holo_file_header_
     */
    void load_fd();

    /*! \brief Open file to initialize raw_rooter_
     */
    void load_footer();
};
} // namespace holovibes::io_files
