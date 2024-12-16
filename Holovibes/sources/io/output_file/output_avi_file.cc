#include "output_avi_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
OutputAviFile::OutputAviFile(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb)
    : OutputFrameFile(file_path)
    , AviFile()
{
    fd_ = fd;
    img_nb_ = img_nb;
    size_length_ = std::max(fd_.width, fd_.height);
}

void OutputAviFile::export_compute_settings(int input_fps, size_t contiguous) {}

void OutputAviFile::write_header()
{
    try
    {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

        cv::Size size = cv::Size(size_length_, size_length_);

        bool is_color = fd_.depth == camera::PixelDepth::Bits24;

        video_writer_ = cv::VideoWriter(file_path_, fourcc, compute_output_fps(), size, is_color);

        if (!video_writer_.isOpened())
            throw cv::Exception();
    }
    catch (const cv::Exception&)
    {
        throw FileException("An error was encountered while trying to create avi file", false);
    }
}

static void move_frame(
    uchar* output, const char* frame, const unsigned short width, const unsigned short height, const int byte_type)
{
    // compute ratio between width and height to know which dimension to fill
    size_t width_ratio = std::max(width / height, 1);
    size_t height_ratio = std::max(height / width, 1);
    boolean add_byte_type = byte_type == camera::Endianness::LittleEndian;
    for (size_t i = 0; i < height * width_ratio; i += width_ratio)
    {
        for (size_t j = 0; j < width * height_ratio; j += height_ratio)
        {
            // depending which dimension is the biggest, we fill the output buffer
            if (width_ratio >= height_ratio)
                for (size_t k = 0; k < width_ratio; k++)
                    output[(i + k) * width + j] =
                        frame[((i / width_ratio) * width + j / height_ratio) * 2 + add_byte_type];
            else
                for (size_t k = 0; k < height_ratio; k++)
                    output[i * width + j + k] =
                        frame[((i / width_ratio) * width + j / height_ratio) * 2 + add_byte_type];
        }
    }
}

size_t OutputAviFile::write_frame(const char* frame, size_t frame_size)
{
    try
    {
        cv::Mat mat_frame;
        bool is_color = fd_.depth == camera::PixelDepth::Bits24;

        if (is_color)
        {
            // this might need to be modified if the frame is not a square
            mat_frame = cv::Mat(size_length_, size_length_, CV_8UC3, const_cast<char*>(frame));
            cv::cvtColor(mat_frame, mat_frame, cv::COLOR_BGR2RGB);
        }

        // else fd_.depth == camera::PixelDepth::Bits16
        else
        {
            // OpenCV does not handle 16 bits video in our case
            // So we make a 8 bits video
            mat_frame = cv::Mat(size_length_, size_length_, CV_8UC1);
            // move frame to the mat_frame.data buffer, and make it square if needed
            move_frame(mat_frame.data, frame, fd_.width, fd_.height, fd_.byteEndian);
        }

        video_writer_ << mat_frame;
    }
    catch (const cv::Exception&)
    {
        throw FileException("Unable to write output avi file frame", false);
    }

    return frame_size;
}

void OutputAviFile::write_footer() {}

void OutputAviFile::correct_number_of_frames(size_t nb_frames_written) { img_nb_ = nb_frames_written; }
} // namespace holovibes::io_files
