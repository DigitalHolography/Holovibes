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

#include "output_file_handler.hh"
#include "output_holo_file.hh"

namespace holovibes::io_files
{
    OutputFile* OutputFileHandler::file_ = nullptr;

    void OutputFileHandler::create(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb)
    {
        if (file_path.ends_with(".holo"))
            file_ = new OutputHoloFile(file_path, fd, img_nb);

        else
            throw FileException("Invalid file extension");
    }

    void OutputFileHandler::close()
    {
        delete file_;
        file_ = nullptr;
    }

    void OutputFileHandler::export_compute_settings(const ComputeDescriptor& cd)
    {
        file_->export_compute_settings(cd);
    }

    void OutputFileHandler::write_header()
    {
        file_->write_header();
    }

    size_t OutputFileHandler::write_frame(char* frame, size_t frame_size)
    {
        return file_->write_frame(frame, frame_size);
    }

    void OutputFileHandler::write_footer()
    {
        file_->write_footer();
    }
} // namespace holovibes::io_files
