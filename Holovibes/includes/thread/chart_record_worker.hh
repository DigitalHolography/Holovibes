/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "worker.hh"

namespace holovibes::worker
{
class ChartRecordWorker : public Worker
{
  public:
    ChartRecordWorker(const std::string& path,
                      const unsigned int nb_frames_to_record);

    void run() override;

  private:
    //! Output record path
    const std::string path_;
    //! Number of points to record
    const unsigned int nb_frames_to_record_;
};
} // namespace holovibes::worker