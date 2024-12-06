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
}

void OutputAviFile::export_compute_settings(int input_fps, size_t contiguous) {}

void OutputAviFile::write_header()
{
    try
    {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

        cv::Size size = cv::Size(fd_.width, fd_.height);

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

size_t OutputAviFile::write_frame(const char* frame, size_t frame_size)
{
    try
    {
        cv::Mat mat_frame;
        bool is_color = fd_.depth == camera::PixelDepth::Bits24;

        if (is_color)
        {
            mat_frame = cv::Mat(fd_.height, fd_.width, CV_8UC3, const_cast<char*>(frame));
            cv::cvtColor(mat_frame, mat_frame, cv::COLOR_BGR2RGB);
        }

        // else fd_.depth == camera::PixelDepth::Bits16
        else
        {
            // OpenCV does not handle 16 bits video in our case
            // So we make a 8 bits video
            mat_frame = cv::Mat(fd_.height, fd_.width, CV_8UC1);

            size_t frame_size_half = frame_size / 2;

            if (fd_.byteEndian == camera::Endianness::LittleEndian)
            {
                for (size_t i = 0; i < frame_size_half; i++)
                    mat_frame.data[i] = frame[2 * i + 1];
            }

            else
            {
                for (size_t i = 0; i < frame_size_half; i++)
                    mat_frame.data[i] = frame[2 * i];
            }
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
