#include "output_frame_file_factory.hh"
#include "output_holo_file.hh"
#include "output_avi_file.hh"
#include "output_moments_avi_file.hh"
#include "output_mp4_file.hh"
#include "output_hdf5_file.hh"

namespace holovibes::io_files
{
OutputFrameFile* OutputFrameFileFactory::create(const std::string& file_path,
                                                const camera::FrameDescriptor& fd,
                                                uint64_t img_nb,
                                                RecordedDataType data_type,
                                                bool moments_avi_fft_shift)
{
    if (file_path.ends_with(".holo"))
        return new OutputHoloFile(file_path, fd, img_nb, data_type);

    else if (file_path.ends_with(".avi"))
    {
        if (data_type == RecordedDataType::MOMENTS)
            return new OutputMomentsAviFile(file_path, fd, moments_avi_fft_shift);
        return new OutputAviFile(file_path, fd, img_nb);
    }

    else if (file_path.ends_with(".mp4"))
        return new OutputMp4File(file_path, fd, img_nb);
    else if (file_path.ends_with(".h5"))
        return new OutputHdf5File(file_path, fd, img_nb);

    else
    {
        std::string msg = "Invalid file extension: " + file_path + ". Supported extensions are: .holo, .avi, .mp4";
        throw FileException(msg, false);
    }
}
} // namespace holovibes::io_files
