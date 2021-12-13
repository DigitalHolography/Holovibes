/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "enum_record_mode.hh"
#include <optional>
#include <array>

namespace holovibes
{
// Fast forward declarations
class Queue;
class ICompute;

namespace worker
{
/*! \class FrameRecordWorker
 *
 * \brief Class used to record frames
 */
class FrameRecordWorker final : public Worker
{
  public:
    /*! \brief Constructor
     *
     * \param file_path The path of the file to record
     * \param nb_frames_to_record The number of frames to record
     * \param record_mode Type of record between raw, holo and cuts
     * \param nb_frames_skip Number of frames to skip before starting
     */
    FrameRecordWorker(const std::string& file_path,
                      std::optional<unsigned int> nb_frames_to_record,
                      RecordMode record_mode,
                      unsigned int nb_frames_skip,
                      unsigned int output_buffer_size);

    void run() override;

  private:
    /*! \brief Init the record queue
     *
     * \return The record queue
     */
    Queue& init_gpu_record_queue();

    /*! \brief Wait for frames to be present in the record queue
     *
     * \param record_queue The record queue
     * \param pipe The compute pipe used to perform the operations
     */
    void wait_for_frames(Queue& record_queue);

    /*! \brief Reset the record queue to free memory
     *
     * \param pipe The compute pipe used to perform the operations
     */
    void reset_gpu_record_queue();

    void integrate_fps_average();

  private:
    /*! \brief The path of the file to record */
    const std::string file_path_;
    /*! \brief The number of frames to record */
    std::optional<unsigned int> nb_frames_to_record_;
    /*! \brief The number of frames to skip before starting the recording */
    unsigned int nb_frames_skip_;
    /*! \brief Whether the raw images are recorded */
    RecordMode record_mode_;
    /*! \brief Output buffer size */
    unsigned int output_buffer_size_;
    /*! \brief Number of frame that keep integrity */
    std::optional<size_t> sure_frames_ = std::nullopt;

    // Average fps is computed with the last 4 values of input fps.
    /*! \brief Useful for Input fps value. */
    unsigned int fps_current_index_ = 0;
    /*! \brief Useful for Input fps value. */
    std::array<unsigned int, 4> fps_buffer_ = {0, 0, 0, 0};

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes
