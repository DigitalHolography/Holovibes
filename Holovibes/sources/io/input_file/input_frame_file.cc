#include "input_frame_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
size_t InputFrameFile::read_frames(char* buffer, size_t frames_to_read, int* flag_packed)
{
    *flag_packed = (frame_size_ / (float)(fd_.width * fd_.height)) * 8;
    long testy = ftell(file_);

    size_t frames_read = std::fread(buffer, frame_size_, frames_to_read, file_);

    if (std::ferror(file_) != 0)
        throw FileException("Unable to read " + std::to_string(frames_to_read) + " frames");

    return frames_read;
}
} // namespace holovibes::io_files
