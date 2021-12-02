/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

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
    ChartRecordWorker(const std::string& path, const unsigned int nb_frames_to_record);

    void run() override;

  private:
    /*! \brief Output record path */
    const std::string path_;
    /*! \brief Number of points to record */
    const unsigned int nb_frames_to_record_;
};
} // namespace holovibes::worker
