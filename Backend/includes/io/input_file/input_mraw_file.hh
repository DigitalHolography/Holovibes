/*! \file
 *
 * \brief Handle loading of .mraw files
 */
#pragma once

#include "input_frame_file.hh"
#include "Mraw_file.hh"

namespace holovibes::io_files
{
/*! \class InputMrawFile
 *
 * \brief Class that represents an mraw file
 */
class InputMrawFile : public InputFrameFile, public MrawFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const override { return MrawFile::get_total_nb_frames(); }

    /*! \brief Set the pointer in the file to the frame requested
     *
     * This method is mandatory to read frames.
     *
     * \throw FileException if an error occurred while setting the position
     */
    void set_pos_to_frame(size_t frame_id) override;
    json import_compute_settings(void) override;
    void import_info(void) const override;

  private:
    // Give access to private members to the factory and the converter
    friend class InputFrameFileFactory;

    /*! \brief Constructor
     *
     * Open the file and read all the required data
     *
     * \param file_path The path of the file to open
     * \throw FileException if an error occurred while opening or reading the file
     */
    InputMrawFile(const std::string& file_path, const std::string& cih);

    /*! \brief Open cih file to initialize  */
    void load_cih(const std::string& cih);
};
} // namespace holovibes::io_files
