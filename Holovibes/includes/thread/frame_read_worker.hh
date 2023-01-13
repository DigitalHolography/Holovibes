/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "queue.hh"
#include "chrono.hh"

namespace holovibes::worker
{
/*! \class FrameReadWorker
 *
 * \brief Abstract class used to read frames
 */
class FrameReadWorker : public Worker
{
  public:
    FrameReadWorker();
    virtual ~FrameReadWorker();

  protected:
    /*! \brief The current fps */
    uint processed_frames_for_fps_;
    uint to_record_;

    /*! \brief Useful for Input fps value. */

    Chrono chrono_;

    float current_display_rate = 30.0f;
    float time_to_wait = 33.0f;

    ImportCache::Cache<> import_cache_;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
