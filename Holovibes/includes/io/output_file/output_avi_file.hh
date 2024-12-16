/*! \file output_avi_file.hh
 *
 * \brief Definition of the OutputAviFile class used to handle the writing of avi files
 */
#pragma once

#include "output_frame_file.hh"
#include "avi_file.hh"
#include "opencv2/opencv.hpp"

namespace holovibes::io_files
{
/*! \class OutputAviFile
 *
 * \brief Class used to handle the writing of avi files
 */
class OutputAviFile : public OutputFrameFile, public AviFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const override { return img_nb_; }

    /*! \brief Export the compute settings in the file
     *
     * \param input_fps how fast the input was
     * \param contiguous How many frames were contiguous in input
     */
    void export_compute_settings(int input_fps, size_t contiguous) override;

    /*! \brief Write the header in the file
     *
     * \throw FileException if an error occurred while writing the header
     */
    void write_header() override;

    /*! \brief Write a frame in the file
     *
     * \param frame The allocated buffer containing the frame
     * \param frame_size The size in bytes of a frame
     * \return The number of bytes written in the file
     * \throw FileException if an error occurred while writing the frame
     */
    size_t write_frame(const char* frame, size_t frame_size) override;

    /*! \brief Write the footer in the file
     *
     * \throw FileException if an error occurred while writing the footer
     */
    void write_footer() override;

    /*! \brief Rewrite the sections in the file where the number of frames has been used
     *
     * It is useful to correct the header when the file is written with a different
     * number of frames than the expected number of frames
     *
     * \throw FileException if an error occurred while correcting the sections
     */
    void correct_number_of_frames(size_t nb_frames_written) override;

  private:
    // Give access to private members to the factory
    friend class OutputFrameFileFactory;

    /*! \brief Constructor
     *
     * Open the file in write only
     *
     * \param file_path The path of the file to open
     * \param fd FrameDescriptor describing the frames of the file to create
     * \param img_nb The number of frames in the file
     * \throw FileException if an error occurred while opening the file
     */
    OutputAviFile(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb);

    /*! \brief The number of images in the file */
    size_t img_nb_;
    /*! \brief The object used to write in the file */
    cv::VideoWriter video_writer_;
    /*! \brief The size of the image side to get a square video */
    unsigned short size_length_;
};
} // namespace holovibes::io_files
