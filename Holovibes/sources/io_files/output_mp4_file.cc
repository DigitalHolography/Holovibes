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

#include "output_mp4_file.hh"
#include "file_exception.hh"

namespace holovibes::io_files
{
    OutputMp4File::OutputMp4File(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb):
        OutputFrameFile(file_path),
        Mp4File()
    {
		fd_ = fd;
        img_nb_ = img_nb;

        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

        try
        {
            bool is_color = fd_.depth == 3;
            video_writer_ = cv::VideoWriter(file_path, fourcc,
                                            20, cv::Size(fd_.width, fd_.height), is_color);

            if (!video_writer_.isOpened())
                throw cv::Exception();
        }
        catch (const cv::Exception&)
        {
            throw FileException("An error was encountered while trying to create mp4 file", false);
        }
    }

    void OutputMp4File::export_compute_settings(const ComputeDescriptor& cd, bool record_raw)
    {}

    void OutputMp4File::write_header()
    {}

    size_t OutputMp4File::write_frame(const char* frame, size_t frame_size)
    {
        try
        {
            cv::Mat mat_frame;
            bool is_color = fd_.depth == 3;

            if (is_color)
            {
                mat_frame = cv::Mat(fd_.height, fd_.width, CV_8UC3, const_cast<char*>(frame));
                cv::cvtColor(mat_frame, mat_frame, cv::COLOR_BGR2RGB);
            }

            // else fd_.depth == 2
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
        catch(const cv::Exception&)
        {
            throw FileException("Unable to write output mp4 file frame", false);
        }

        return frame_size;
    }

    void OutputMp4File::write_footer()
    {}

	void OutputMp4File::correct_number_of_frames(unsigned int nb_frames_written)
	{}
} // namespace holovibes::io_files
