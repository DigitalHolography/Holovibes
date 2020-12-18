/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "output_frame_file.hh"
#include "mp4_file.hh"
#include "opencv2/opencv.hpp"

namespace holovibes::io_files
{
class OutputMp4File : public OutputFrameFile, public Mp4File
{
  public:
    /*!
     *  \brief    Getter on the total number of frames in the file
     */
    size_t get_total_nb_frames() const override;

    /*!
     *  \brief    Export the compute settings in the file
     *
     *  \param    cd            The ComputeDescriptor containing the compute
     * settings \param    record_raw    Is the raw record enabled
     */
    void export_compute_settings(const ComputeDescriptor& cd,
                                 bool record_raw) override;

    /*!
     *  \brief    Write the header in the file
     *
     *  \throw    FileException if an error occurred while writing the header
     */
    void write_header() override;

    /*!
     *  \brief    Write a frame in the file
     *
     *  \param    frame        The allocated buffer containing the frame
     *  \param    frame_size   The size in bytes of a frame
     *
     *  \return   The number of bytes written in the file
     *
     *  \throw    FileException if an error occurred while writing the frame
     */
    size_t write_frame(const char* frame, size_t frame_size) override;

    /*!
     *  \brief    Write the footer in the file
     *
     *  \throw    FileException if an error occurred while writing the footer
     */
    void write_footer() override;

    /*!
     *  \brief    Rewrite the sections in the file where the number of frames
     * has been used \details  It is useful to correct the header when the file
     * is written with a different number of frames than the expected number of
     * frames
     *
     *  \throw    FileException if an error occurred while correcting the
     * sections
     */
    void correct_number_of_frames(size_t nb_frames_written) override;

  private:
    // Give access to private members to the factory
    friend class OutputFrameFileFactory;

    /*!
     *  \brief    Constructor
     *
     *  \details  Open the file in write only
     *
     *  \param    file_path    The path of the file to open
     *  \param    fd           FrameDescriptor describing the frames of the file
     * to create \param    img_nb       The number of frames in the file
     *
     *  \throw    FileException if an error occurred while opening the file
     */
    OutputMp4File(const std::string& file_path,
                  const camera::FrameDescriptor& fd,
                  uint64_t img_nb);

    //! The number of images in the file
    size_t img_nb_;
    //! The object used to write in the file
    cv::VideoWriter video_writer_;
};
} // namespace holovibes::io_files

#include "output_mp4_file.hxx"