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

#include "output_file.hh"
#include "holo_file.hh"

namespace holovibes::io_files
{
    class OutputHoloFile: public OutputFile, public HoloFile
    {
    public:
        /*!
         *  \brief    Getter on the total number of frames in the file
         */
        size_t get_total_nb_frames() const override;

        /*!
         *  \brief    Export the compute settings in the file
         *
         *  \param    cd    The ComputeDescriptor containing the compute settings
         */
        void export_compute_settings(const ComputeDescriptor& cd) override;

        /*!
         *  \brief    Write the header in the file
         *
         *  \throw    FileException if an error occurred while writing the header
         */
        void write_header() override;

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
        size_t write_frame(const char* frame, size_t frame_size) override;

        /*!
         *  \brief    Write the footer in the file
         *
         *  \throw    FileException if an error occurred while writing the footer
         */
        void write_footer() override;

    private:
        // Give access to private members to the handler
        friend class OutputFileHandler;

        /*!
         *  \brief    Constructor
         *
         *  \details  Open the file in write only
         *
         *  \param    file_path    The path of the file to open
         *  \param    fd           FrameDescriptor describing the frames of the file to create
         *  \param    img_nb       The number of frames in the file
         *
         *  \throw    FileException if an error occurred while opening the file
         */
        OutputHoloFile(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb);
    };
} // namespace holovibes::io_files

#include "output_holo_file.hxx"