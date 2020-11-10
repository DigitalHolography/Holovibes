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

#include "io_file.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
    /*!
     *  \brief    Base class of Output files
     */
    class OutputFile: public IOFile
    {
    public:
        /*!
         *  \brief    Export the compute settings in the file
         *
         *  \param    cd    The ComputeDescriptor containing the compute settings
         */
        virtual void export_compute_settings(const ComputeDescriptor& cd) = 0;

        /*!
         *  \brief    Write the header in the file
         *
         *  \throw    FileException if an error occurred while writing the header
         */
        virtual void write_header() = 0;

        // FIXME: update write_frame method, try to remove frame_size
        /*!
         *  \brief    Write a frame in the file
         *
         *  \param    frame        The allocated buffer containing the frame
         *  \param    frame_size   The size in bytes of a frame
         *
         *  \return   The number of bytes written in the file
         *
         *  \throw    FileException if an error occurred while writing the frame
         */
        virtual size_t write_frame(const char* frame, size_t frame_size) = 0;

        /*!
         *  \brief    Write the footer in the file
         *
         *  \throw    FileException if an error occurred while writing the footer
         */
        virtual void write_footer() = 0;

    protected:
        // Give access to protected members to the handler
        friend class OutputFileHandler;

        /*!
         *  \brief    Constructor
         *
         *  \details  Open the file in write only
         *
         *  \param    file_path    The path of the file to open
         *
         *  \throw    FileException if an error occurred while opening the file
         */
        OutputFile(const std::string& file_path);
    };
} // namespace holovibes::io_files

#include "output_file.hxx"
