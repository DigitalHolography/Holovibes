/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "export_cache.hh"
#include "env_structs.hh"

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
     * \param record_mode Type of record between raw, holo and cuts
     * \param nb_frames_skip Number of frames to skip before starting
     */
    FrameRecordWorker();

    void run() override;

  private:
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

  private:
    // Average fps is computed with the last FPS_LAST_X_VALUES values of input fps.
    /*! \brief Useful for Input fps value. */
    unsigned int fps_current_index_ = 0;
    /*! \brief Useful for Input fps value. */
    std::array<unsigned int, FPS_LAST_X_VALUES> fps_buffer_ = {0};

    /*! \brief Integrate Input Fps in fps_buffers if relevent */
    void integrate_fps_average();
    /*! \brief Compute fps_buffer_ average on the correct number of value */
    size_t compute_fps_average() const;

     uint current_fps_;
     uint processed_fps_;
     
    FrameRecordEnv& env_;
    ExportCache::Cache<> export_cache_;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
