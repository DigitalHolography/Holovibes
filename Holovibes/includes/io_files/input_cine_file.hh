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

#include "input_file.hh"
#include "cine_file.hh"
#include "compute_descriptor.hh"

namespace holovibes::io_files
{
    class InputCineFile: public InputFile, public CineFile
    {
    public:
        /*!
         *  \brief    Getter on the total number of frames in the file
         */
        size_t get_total_nb_frames() const override;

        /*!
         *  \brief    Set the pointer in the file to the first frame
         *  \details  This method is mandatory to read frames.
         *
         *  \throw    FileException if an error occurred while setting the position
         */
        void set_pos_to_first_frame() override;

        /*!
         *  \brief    Update ComputeDescriptor with the settings present in the file
         *
         *  \param    cd    The ComputeDescriptor to update
         */
        void import_compute_settings(ComputeDescriptor& cd) const override;

    private:
        // Give access to private members to the handler
        friend class InputFileHandler;

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
