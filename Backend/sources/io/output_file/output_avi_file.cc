#include "output_avi_file.hh"
#include "API.hh"
#include "file_exception.hh"
#include <algorithm>
#include <cmath>

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
        cv::Size size;
        if (API.transform.get_space_transformation() == SpaceTransformation::FRESNELTR)
            size = cv::Size(size_length_, size_length_);
        else
            size = cv::Size(fd_.width, fd_.height);

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
// Bi-linear interpolation
static void
move_frame(uchar* output, const char* frame, const ushort input_width, const ushort input_height, const int byte_type)
{
    ushort output_size = std::max(input_width, input_height);

    float x_ratio = static_cast<float>(input_width) / output_size;
    float y_ratio = static_cast<float>(input_height) / output_size;

    bool add_byte_type = byte_type == camera::Endianness::LittleEndian;

    for (size_t y = 0; y < output_size; ++y)
    {
        for (size_t x = 0; x < output_size; ++x)
        {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x1 = static_cast<int>(std::floor(src_x));
            int y1 = static_cast<int>(std::floor(src_y));
            int x2 = std::min(x1 + 1, static_cast<int>(input_width - 1));
            int y2 = std::min(y1 + 1, static_cast<int>(input_height - 1));

            float x_diff = src_x - x1;
            float y_diff = src_y - y1;

            uchar pixel11 = static_cast<uchar>(frame[(y1 * input_width + x1) * 2 + add_byte_type]);
            uchar pixel12 = static_cast<uchar>(frame[(y1 * input_width + x2) * 2 + add_byte_type]);
            uchar pixel21 = static_cast<uchar>(frame[(y2 * input_width + x1) * 2 + add_byte_type]);
            uchar pixel22 = static_cast<uchar>(frame[(y2 * input_width + x2) * 2 + add_byte_type]);

            float value = (pixel11 * (1 - x_diff) * (1 - y_diff)) + (pixel12 * x_diff * (1 - y_diff)) +
                          (pixel21 * (1 - x_diff) * y_diff) + (pixel22 * x_diff * y_diff);

            output[y * output_size + x] = static_cast<uchar>(value);
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
            // move frame to the mat_frame.data buffer, and make it square if needed
            if (API.transform.get_space_transformation() == SpaceTransformation::FRESNELTR)
            {
                mat_frame = cv::Mat(size_length_, size_length_, CV_8UC1);
                move_frame(mat_frame.data, frame, fd_.width, fd_.height, fd_.byteEndian);
            }
            else
            {
                // if the space transform isn't FRESNELTR there is no interpolation of the output
                mat_frame = cv::Mat(fd_.height, fd_.width, CV_8UC1);
                const ushort* src = reinterpret_cast<const ushort*>(frame);
                uchar* dest = mat_frame.data;

                for (int y = 0; y < fd_.height; ++y)
                {
                    const ushort* src_row = src + y * fd_.width;
                    uchar* dest_row = dest + y * fd_.width;

                    for (int x = 0; x < fd_.width; ++x)
                        dest_row[x] = static_cast<uchar>(src_row[x] >> 8);
                }
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
