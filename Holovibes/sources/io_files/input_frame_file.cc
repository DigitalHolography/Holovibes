#include "input_frame_file.hh"
#include "file_exception.hh"
#include "logger.hh"

namespace holovibes::io_files
{
size_t InputFrameFile::read_frames(char* buffer, size_t frames_to_read, int* flag_packed)
{
    if (std::ferror(file_) != 0)
        throw FileException("HOLO: Error on check before reading " + std::to_string(frames_to_read) +
                            " frames with code : " + std::to_string(std::ferror(file_)));

    *flag_packed = (frame_size_ / (float)(fd_.width * fd_.height)) * 8;
    size_t frames_read = std::fread(buffer, frame_size_, frames_to_read, file_);

    if (frames_read != frames_to_read)
    {
        LOG_DEBUG("Couldn't read all needed frames : read {} instead of {}", frames_read, frames_to_read);
        frames_read = std::fread(buffer, frame_size_, 1, file_);

        if (frames_read != 1)
        {
            LOG_DEBUG("Couldn't read any images from this file {}", std::ftell(file_));
        }
    }

    if (std::ferror(file_) != 0)
        throw FileException("HOLO: Error while reading " + std::to_string(frames_to_read) +
                            " frames, with code : " + std::to_string(std::ferror(file_)));

    return frames_read;
}
} // namespace holovibes::io_files
