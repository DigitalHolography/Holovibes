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

#include "file_exception.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
    class OutputFile;

    /*!
     *  \brief    Used to do the interface between the file system and the other classes
     *
     *  \details  This class owns the current output file and implements static methods to manipulate it
     */
    class OutputFileHandler
    {
    public:
        /*!
         *  \brief    Deleted default constructor
         */
        OutputFileHandler() = delete;

        /*!
         *  \brief    Deleted default destructor
         */
        ~OutputFileHandler() = delete;

        /*!
         *  \brief    Deleted default copy constructor
         */
        OutputFileHandler(const OutputFileHandler&) = delete;

        /*!
         *  \brief    Deleted default copy operator
         */
        OutputFileHandler& operator=(const OutputFileHandler&) = delete;

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
         *  \throw    FileException if the OutputFile is not created
         *            or if the file extension is not supported
         */
        static void create(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb);

        /*!
         *  \brief    Close the current output file
         *
         *  \details  If there is no output file, do nothing
         */
        static void close();

        /*!
         *  \brief    Export the compute settings in the current output file
         *
         *  \param    cd    The ComputeDescriptor containing the compute settings
         */
        static void export_compute_settings(const ComputeDescriptor& cd);

        /*!
         *  \brief    Write the header in the current output file
         *
         *  \throw    FileException if an error occurred while writing the header
         */
        static void write_header();

        /*!
         *  \brief    Write a frame in the current output file
         *
         *  \param    frame        The allocated buffer containing the frame
         *  \param    frame_size   The size in bytes of a frame
         *
         *  \return   The number of bytes written in the file
         *
         *  \throw    FileException if an error occurred while writing the frame
         */
        static size_t write_frame(char* frame, size_t frame_size);

        /*!
         *  \brief    Write the footer in the current output file
         *
         *  \throw    FileException if an error occurred while writing the footer
         */
        static void write_footer();

    private:
        //! Current output file
        static OutputFile* file_;
    };
} // namespace holovibes::io_files
