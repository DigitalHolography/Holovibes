/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "input_frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
/*!
 *  \brief    Used to open an input file
 *
 *  \details  This class is a factory,
 *            the created input file depends on the file path extension
 */
class InputFrameFileFactory
{
  public:
    /*!
     *  \brief    Deleted default constructor
     */
    InputFrameFileFactory() = delete;

    /*!
     *  \brief    Deleted default destructor
     */
    ~InputFrameFileFactory() = delete;

    /*!
     *  \brief    Deleted default copy constructor
     */
    InputFrameFileFactory(const InputFrameFileFactory&) = delete;

    /*!
     *  \brief    Deleted default copy operator
     */
    InputFrameFileFactory& operator=(const InputFrameFileFactory&) = delete;

    /*!
     *  \brief    Open an input file
     *
     *  \details  This methods allocates the input file attribute.
     *            Thus, it must be called before the other methods
     *
     *
     *  \param    file_path    The path of the file to open,
     *                         the extension must be supported
     *
     *  \return   Pointer on the opened input file
     *
     *  \throw    FileException if the input file is invalid,
     *            if there is an error while opening the file
     *            or if the file extension is not supported
     */
    static io_files::InputFrameFile* open(const std::string& file_path);
};
} // namespace holovibes::io_files
