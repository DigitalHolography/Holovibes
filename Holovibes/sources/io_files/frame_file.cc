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

#include "frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
    FrameFile::FrameFile(const std::string& file_path, FrameFile::OpeningMode mode)
    {
        if (mode == FrameFile::OpeningMode::READ)
            file_ = fopen(file_path.c_str(), "rb");

        else
            file_ = fopen(file_path.c_str(), "wb");

        // if an error occurred
        if (file_ == nullptr)
            throw FileException("Unable to open file " + file_path + ": " + std::strerror(errno));
    }

    FrameFile::~FrameFile()
    {
        std::fclose(file_);
    }
} // namespace holovibes::io_files
