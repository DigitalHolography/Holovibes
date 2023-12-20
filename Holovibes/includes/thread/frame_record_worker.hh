/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "enum_record_mode.hh"
#include <optional>
#include <array>

#define FPS_LAST_X_VALUES 16

namespace holovibes
{
// Fast forward declarations
class Queue;
class ICompute;
} // namespace holovibes

namespace holovibes::worker
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
     * \param nb_frames_skip Number of frames to skip before starting
     */
    FrameRecordWorker(const std::string& file_path,
                      std::optional<unsigned int> nb_frames_to_record,
                      unsigned int nb_frames_skip,
                      std::atomic<std::shared_ptr<Queue>>& record_queue);

    void run() override;

  private:
    /*! \brief Init the record queue
     *
     * \return The record queue
     */
    // Queue& init_record_queue();

    /*! \brief Wait for frames to be present in the record queue*/
    void wait_for_frames();

    /*! \brief Reset the record queue to free memory
     *
     * \param pipe The compute pipe used to perform the operations
     */
    void reset_record_queue();

  private:
    /*! \brief The path of the file to record */
    const std::string file_path_;
    /*! \brief The number of frames to record */
    std::optional<unsigned int> nb_frames_to_record_;
    /*! \brief The number of frames to skip before starting the recording */
    unsigned int nb_frames_skip_;

    // Average fps is computed with the last FPS_LAST_X_VALUES values of input fps.
    /*! \brief Useful for Input fps value. */
    unsigned int fps_current_index_ = 0;
    /*! \brief Useful for Input fps value. */
    std::array<unsigned int, FPS_LAST_X_VALUES> fps_buffer_ = {0};

    /*! \brief Integrate Input Fps in fps_buffers if relevent */
    void integrate_fps_average();
    /*! \brief Compute fps_buffer_ average on the correct number of value */
    size_t compute_fps_average() const;

    const cudaStream_t stream_;

    /*! \brief The queue in which the frames are stored for record*/
    std::atomic<std::shared_ptr<Queue>>& record_queue_;
};
} // namespace holovibes::worker
