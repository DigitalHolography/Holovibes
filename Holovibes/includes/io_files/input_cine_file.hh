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
#include "cine_file.hh"

namespace holovibes::io_files
{
    // FIXME: this class could and should be improved.
    // It handles the cine files like the previous versions of Holovibes,
    // but it does not handle all the specificities of the cine files (see link in cine_file.hh).
    class InputCineFile: public InputFrameFile, public CineFile
    {
    public:
        /*!
         *  \brief    Getter on the total number of frames in the file
         */
        size_t get_total_nb_frames() const override;

        /*!
         *  \brief    Update ComputeDescriptor with the settings present in the file
         *
         *  \param    cd    The ComputeDescriptor to update
         */
        void import_compute_settings(ComputeDescriptor& cd) const override;

        /*!
         *  \brief    Set the pointer in the file to the frame requested
         *  \details  This method is mandatory to read frames.
         *
         *  \throw    FileException if an error occurred while setting the position
         */
        void set_pos_to_frame(size_t frame_id) override;

        /*!
         *  \brief    Read frames in the file
         *
         *  \param    buffer            The allocated buffer in which the frames should be stored
         *  \param    frames_to_read    The number of frames to read
         *
         *  \return   The actual number of frames read
         *
         *  \throw    FileException if an error occurred while reading the file
         */
        size_t read_frames(char* buffer, size_t frames_to_read) override;

    private:
        // Give access to private members to the factory
        friend class InputFrameFileFactory;

        /*!
         *  \brief    Constructor
         *
         *  \details  Open the file and read all the required data
         *
         *  \param    file_path    The path of the file to open
         *
         *  \throw    FileException if an error occurred while opening or reading the file
         */
        InputCineFile(const std::string& file_path);
    };
} // namespace holovibes::io_files

#include "input_cine_file.hxx"
