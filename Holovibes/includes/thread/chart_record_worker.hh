/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "env_structs.hh"
#include "export_cache.hh"

namespace holovibes::worker
{
/*! \class ChartRecordWorker
 *
 * \brief Class used to record chart
 */
class ChartRecordWorker final : public Worker
{
  public:
    /*!
     * \param path Output record path
     * \param nb_frames_to_record Number of points to record
     */
    ChartRecordWorker();

    void run() override;

  private:
    // FIXME API : this must be by value instead of a pipe reference
    ChartEnv& env_;
    ExportCache::Cache<> export_cache_;
};
} // namespace holovibes::worker
