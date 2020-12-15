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

#include "output_frame_file.hh"

namespace holovibes::io_files
{
    inline OutputFrameFile::OutputFrameFile(const std::string& file_path): FrameFile(file_path, FrameFile::OpeningMode::WRITE)
    {}

    inline void OutputFrameFile::set_make_square_output(bool make_square_output)
    {
        // if the output is anamorphic and should be a square output
        if (make_square_output && fd_.width != fd_.height)
            max_side_square_output_ = std::max(fd_.width, fd_.height);
    }
} // namespace holovibes::io_files
