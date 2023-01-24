#include "output_frame_file_factory.hh"
#include "output_holo_file.hh"
#include "output_avi_file.hh"
#include "output_mp4_file.hh"

namespace holovibes::io_files
{
OutputFrameFile*
OutputFrameFileFactory::create(const std::string& file_path, const FrameDescriptor& fd, uint64_t img_nb)
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
