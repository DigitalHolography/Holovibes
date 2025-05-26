/*! \file
 *
 * \brief Definition of the InputFrameFileFactory class.
 */
#pragma once

#include "input_frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
/*! \class InputFrameFileFactory
 *
 * \brief Used to open an input file
 *
 * This class is a factory, the created input file depends on the file path extension
 */
class InputFrameFileFactory
{
  public:
    /*! \brief Deleted default constructor */
    InputFrameFileFactory() = delete;

    /*! \brief Deleted default destructor */
    ~InputFrameFileFactory() = delete;

    /*! \brief Deleted default copy constructor */
    InputFrameFileFactory(const InputFrameFileFactory&) = delete;

    /*! \brief Deleted default copy operator */
    InputFrameFileFactory& operator=(const InputFrameFileFactory&) = delete;

    /*! \brief Open an input file
     *
     * This methods allocates the input file attribute.
     * Thus, it must be called before the other methods
     *
     * \param file_path The path of the file to open, the extension must be supported
     * \return Pointer on the opened input file
     * \throw FileException if the input file is invalid, if there is an error while opening the file
     * or if the file extension is not supported
     */
    static io_files::InputFrameFile* open(const std::string& file_path);

    /*! \brief Open an mraw input file
     *
     * This methods allocates the input file attribute.
     * Thus, it must be called before the other methods
     *
     * \param file_path The path of the .mraw file to open.
     * \param cih The path of the .cih file needed to open the mraw.
     * \return Pointer on the opened input file
     * \throw FileException if the input file is invalid, if there is an error while opening the file
     * or if the file extension is not supported
     */
    static io_files::InputFrameFile* open_mraw(const std::string& file_path, const std::string& cih);
};
} // namespace holovibes::io_files
