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

#include "output_frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
    class OutputFrameFile;

    /*!
     *  \brief    Used to create an output file
     *
     *  \details  This class is a factory,
     *            the created input file depends on the file path extension
     */
    class OutputFrameFileFactory
    {
    public:
        /*!
         *  \brief    Deleted default constructor
         */
        OutputFrameFileFactory() = delete;

        /*!
         *  \brief    Deleted default destructor
         */
        ~OutputFrameFileFactory() = delete;

        /*!
         *  \brief    Deleted default copy constructor
         */
        OutputFrameFileFactory(const OutputFrameFileFactory&) = delete;

        /*!
         *  \brief    Deleted default copy operator
         */
        OutputFrameFileFactory& operator=(const OutputFrameFileFactory&) = delete;

        /*!
         *  \brief    Create an output file
         *
         *  \details  This methods allocates the output file attribute.
         *            Thus, it must be called before the other methods
         *
         *
         *  \param    file_path    The path of the file to create,
         *                         the extension must be supported
         *
         *  \throw    FileException if the OutputFrameFile is not created
         *            or if the file extension is not supported
         */
        static OutputFrameFile* create(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb);
    };
} // namespace holovibes::io_files
