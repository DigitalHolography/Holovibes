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
#include "frame_desc.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
    class InputFile;

    /*!
     *  \brief    Used to do the interface between the file system and the other classes
     *
     *  \details  This class owns the current input file and implements static methods to manipulate it
     */
    class InputFileHandler
    {
    public:
        /*!
         *  \brief    Deleted default constructor
         */
        InputFileHandler() = delete;

        /*!
         *  \brief    Deleted default destructor
         */
        ~InputFileHandler() = delete;

        /*!
         *  \brief    Deleted default copy constructor
         */
        InputFileHandler(const InputFileHandler&) = delete;

        /*!
         *  \brief    Deleted default copy operator
         */
        InputFileHandler& operator=(const InputFileHandler&) = delete;

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
         *  \throw    FileException if the InputFile is not created
         *            or if the file extension is not supported
         */
        static void open(const std::string& file_path);

        /*!
         *  \brief    Close the current input file
         *
         *  \details  If there is no input file, do nothing
         */
        static void close();

        /*!
         *  \brief    Get the frame descriptor of the current input file
         *
         *  \return   FrameDescriptor of the current input file
         */
        static const camera::FrameDescriptor& get_frame_descriptor();

        /*!
         *  \brief    Get the total number of frames in the current input file
         *
         *  \return   Number of frames in the current input file
         */
        static size_t get_total_nb_frames();

        /*!
         *  \brief    Get the frame annotation size of the current input file
         *
         *  \details  I.e. the number of bytes present before each frame
         *
         *  \return   The size of a frame annotation
         */
        static size_t get_frame_annotation_size();

        /*!
         *  \brief    Update compute settings with settings present in the current input file
         *
         *  \param    cd    The ComputeDescriptor in which update the settings
         */
        static void import_compute_settings(ComputeDescriptor& cd);

        /*!
         *  \brief    Set position in the current input file to the first frame
         *
         *  \details  This method must be called before the first read_frames call
         *
         *  \throw    FileException if an error occurred while setting the position
         */
        static void set_pos_to_first_frame();

        /*!
         *  \brief    Read frames in the current input file
         *
         *  \param    buffer            The allocated buffer in which the frames should be stored
         *  \param    frames_to_read    The number of frames to read
         *
         *  \return   The actual number of frames read
         *
         *  \throw    FileException if an error occurred while reading the file
         */
        static size_t read_frames(char* buffer, size_t frames_to_read);

    private:
        //! Current input file
        static InputFile* file_;
    };
} // namespace holovibes::io_files
