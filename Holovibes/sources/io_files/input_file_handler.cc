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

#include "input_file_handler.hh"
#include "input_holo_file.hh"
#include "input_cine_file.hh"

namespace holovibes::io_files
{
    InputFile* InputFileHandler::file_ = nullptr;

    void InputFileHandler::open(const std::string& file_path)
    {
        if (file_path.ends_with(".holo"))
            file_ = new InputHoloFile(file_path);

        else if (file_path.ends_with(".cine"))
            file_ = new InputCineFile(file_path);

        else
            throw FileException("Invalid file extension");
    }

    void InputFileHandler::close()
    {
        delete file_;
        file_ = nullptr;
    }

    const camera::FrameDescriptor& InputFileHandler::get_frame_descriptor()
    {
        return file_->get_frame_descriptor();
    }

    size_t InputFileHandler::get_total_nb_frames()
    {
        return file_->get_total_nb_frames();
    }

    size_t InputFileHandler::get_frame_annotation_size()
    {
        return file_->get_frame_annotation_size();
    }

    void InputFileHandler::import_compute_settings(holovibes::ComputeDescriptor& cd)
    {
        file_->import_compute_settings(cd);
    }

    void InputFileHandler::set_pos_to_first_frame()
    {
        file_->set_pos_to_first_frame();
    }

    size_t InputFileHandler::read_frames(char* buffer, size_t frames_to_read)
    {
        return file_->read_frames(buffer, frames_to_read);
    }
} // namespace holovibes::io_files
