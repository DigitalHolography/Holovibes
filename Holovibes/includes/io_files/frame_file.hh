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

#include "frame_desc.hh"

namespace holovibes::io_files
{
    /*!
     *  \brief    Base class of Input and Output files
     *
     *  \details  Some methods, such as the constructors and the destructors,
     *            are protected to prevent the developers to access a file
     *            outside the file system.
     *            An UML diagram, called file_system, is provided in
     *            the documentation folder
     */
    class FrameFile
    {
    public:
        /*!
         *  \brief    Getter on the frame descriptor of the file
         */
        camera::FrameDescriptor get_frame_descriptor();

        /*!
         *  \brief    Getter on the frame descriptor of the file
         */
        const camera::FrameDescriptor& get_frame_descriptor() const;

        /*!
         *  \brief    Getter on the total number of frames in the file
         */
        virtual size_t get_total_nb_frames() const = 0;

    protected:
        /*!
         *  \brief    Enum representing the opening mode of a file
         */
        enum class OpeningMode
        {
            READ,
            WRITE
        };

        /*!
         *  \brief    Constructor
         *
         *  \details  Open the file with the desired mode
         *
         *  \param    file_path    The path of the file to open
         *  \param    mode         The opening mode of the file
         *
         *  \throw    FileException if an error occurred while opening the file
         */
        FrameFile(const std::string& file_path, OpeningMode mode);

        /*!
         *  \brief    Destructor
         *
         *  \details  Close the file
         */
        virtual ~FrameFile();

        /*!
         *  \brief    Default copy constructor
         */
        FrameFile(const FrameFile&) = default;

        /*!
         *  \brief    Default copy operator
         */
        FrameFile& operator=(const FrameFile&) = default;

        //! Frame descriptor associated to the file
        camera::FrameDescriptor fd_;
        //! Pointer associated to the file
        //! C way because it is faster
        std::FILE* file_ = nullptr;
    };
} // namespace holovibes::io_files

#include "frame_file.hxx"
