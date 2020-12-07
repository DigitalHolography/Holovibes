/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
