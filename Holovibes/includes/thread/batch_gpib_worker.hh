/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <deque>

#include "gpib_dll.hh"
#include "IVisaInterface.hh"
#include "gpib_controller.hh"
#include "gpib_exceptions.hh"

#include "enum_record_mode.hh"

#include "frame_record_worker.hh"
#include "chart_record_worker.hh"

namespace holovibes::worker
{
class BatchGPIBWorker : public Worker
{
  public:
    BatchGPIBWorker(const std::string& batch_input_path,
                    const std::string& output_path,
                    unsigned int nb_frames_to_record,
                    RecordMode record_mode,
                    bool square_output);

    void stop() override;

    void run() override;

  private:
    void parse_file(const std::string& batch_input_path);

    void execute_instrument_command(gpib::BatchCommand instrument_command);

    std::string format_batch_output(const unsigned int index);

  private:
    const std::string output_path_;

    const unsigned int nb_frames_to_record_;

    const RecordMode record_mode_;

    const bool square_output_;

    std::unique_ptr<FrameRecordWorker> frame_record_worker_;

    std::unique_ptr<ChartRecordWorker> chart_record_worker_;

    std::deque<gpib::BatchCommand> batch_cmds_;

    std::shared_ptr<gpib::IVisaInterface> gpib_interface_;
};
} // namespace holovibes::worker