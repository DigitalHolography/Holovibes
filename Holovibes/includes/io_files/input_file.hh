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
     *  \brief    Base class of Input files
     */
    class InputFile: public IOFile
    {
    public:
        /*!
         *  \brief    Getter on the size of a frame annotation in the file
         */
        size_t get_frame_annotation_size() const;

        /*!
         *  \brief    Update ComputeDescriptor with the settings present in the file
         *
         *  \param    cd    The ComputeDescriptor to update
         */
        virtual void import_compute_settings(ComputeDescriptor& cd) const = 0;

        /*!
         *  \brief    Set the pointer in the file to the first frame
         *  \details  This method must be called before the first read_frames call
         *
         *  \throw    FileException if an error occurred while setting the position
         */
        virtual void set_pos_to_first_frame() = 0;

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
        size_t read_frames(char* buffer, size_t frames_to_read);

    protected:
        // Give access to protected members to the handler
        friend class InputFileHandler;

        /*!
         *  \brief    Constructor
         *
         *  \details  Open the file in read only
         *
         *  \param    file_path    The path of the file to open
         *
         *  \throw    FileException if an error occurred while opening the file
         */
        InputFile(const std::string& file_path);

        //! The frame annotation size, i.e. the number of bytes present before each frame
        size_t frame_annotation_size_;
        //! The actual frame size, i.e. the frame size plus the frame annotation size
        size_t actual_frame_size_;
    };
} // namespace holovibes::io_files

#include "input_file.hxx"
