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
    FrameRecordWorker();
    ~FrameRecordWorker();

    void run() override;

  private:
    /*! \brief Wait for frames to be present in the record queue
     *
     * \param record_queue The record queue
     * \param pipe The compute pipe used to perform the operations
     */
    void wait_for_frames(Queue& record_queue);

    const cudaStream_t& get_stream() const { return stream_; }

  private:
    FrameRecordEnv& env_;
    ExportCache::Cache<> export_cache_;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
