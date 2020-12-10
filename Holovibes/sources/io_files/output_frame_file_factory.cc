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

#include "output_frame_file_factory.hh"
#include "output_holo_file.hh"
#include "output_avi_file.hh"
#include "output_mp4_file.hh"

namespace holovibes::io_files
{
    OutputFrameFile* OutputFrameFileFactory::create(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb)
    {
        if (file_path.ends_with(".holo"))
            return new OutputHoloFile(file_path, fd, img_nb);

        else if (file_path.ends_with(".avi"))
            return new OutputAviFile(file_path, fd, img_nb);

        else if (file_path.ends_with(".mp4"))
            return new OutputMp4File(file_path, fd, img_nb);

        else
            throw FileException("Invalid file extension", false);
    }
} // namespace holovibes::io_files
