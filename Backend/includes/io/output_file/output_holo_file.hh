/*! \file output_holo_file.hh
 *
 * \brief Defines the OutputHoloFile class, responsible for exporting data to a .holo file. The saved file can be
 * reopend with the Holovibes application and additional computation can be performed on it.
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "output_frame_file.hh"
#include "holo_file.hh"
#include "enum_recorded_data_type.hh"

namespace holovibes::io_files
{
/*! \brief Per-frame timestamps stored while recording, kept compact in the recording hot path. */
struct FrameTimestampUs
{
    uint64_t unix_us;
    uint64_t camera_us;
    uint64_t offset_us;
};

static_assert(sizeof(FrameTimestampUs) == 3 * sizeof(uint64_t));

/*! \class OutputHoloFile
 *
 * \brief Class responsible for exporting data to a .holo file. The saved file can be reopend with Holovibes since both
 * images and computation settings are saved.
 */
class OutputHoloFile : public OutputFrameFile, public HoloFile
{
  public:
    /*! \brief Getter on the total number of frames in the file */
    size_t get_total_nb_frames() const override { return HoloFile::get_total_nb_frames(); }

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
     * It is useful to correct the header when the file is written with a
     * different number of frames than the expected number of frames
     *
     * \throw FileException if an error occurred while correcting the sections
     */
    void correct_number_of_frames(size_t nb_frames_written) override;

    /*! \brief Set the timestamps of the first and last frames of the session
     *
     * \param first_us Timestamp in microseconds of the first frame
     * \param last_us Timestamp in microseconds of the last frame
     * \param first_camera_us Camera timestamp in microseconds of the first frame
     * \param last_camera_us Camera timestamp in microseconds of the last frame
     * \param offset_us Offset in microseconds applied to the camera timestamps to obtain the system timestamps
     */
    void set_session_timestamps_us(uint64_t first_us,
                                   uint64_t last_us,
                                   uint64_t first_camera_us,
                                   uint64_t last_camera_us,
                                   uint64_t first_offset_us,
                                   uint64_t last_offset_us)
    {
        session_first_ts_us_ = first_us;
        session_last_ts_us_ = last_us;
        session_first_camera_ts_us_ = first_camera_us;
        session_last_camera_ts_us_ = last_camera_us;
        session_first_offset_us_ = first_offset_us;
        session_last_offset_us_ = last_offset_us;
        has_session_ts_ = true;
    }

    /*! \brief Move the recorded per-frame timestamps into the output file metadata builder. */
    void set_frame_timestamps_us(std::vector<FrameTimestampUs>&& timestamps) noexcept
    {
        session_frame_timestamps_us_ = std::move(timestamps);
    }

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
    OutputHoloFile(const std::string& file_path,
                   const camera::FrameDescriptor& fd,
                   uint64_t img_nb,
                   RecordedDataType data_type);

    bool has_session_ts_ = false;
    uint64_t session_first_ts_us_ = 0;
    uint64_t session_last_ts_us_ = 0;
    uint64_t session_first_camera_ts_us_ = 0;
    uint64_t session_last_camera_ts_us_ = 0;
    uint64_t session_first_offset_us_ = 0;
    uint64_t session_last_offset_us_ = 0;
    std::vector<FrameTimestampUs> session_frame_timestamps_us_;
};
} // namespace holovibes::io_files
