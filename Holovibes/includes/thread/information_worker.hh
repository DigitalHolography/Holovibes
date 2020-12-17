/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "worker.hh"

namespace holovibes
{
class InformationContainer;

namespace worker
{
class InformationWorker : public Worker
{
  public:
    InformationWorker(bool is_cli, InformationContainer& info);

    void run() override;

  private:
    void compute_fps(long long waited_time);

    void compute_throughput(ComputeDescriptor& cd,
                            unsigned int output_frame_res,
                            unsigned int input_frame_size,
                            unsigned int record_frame_size);

    void display_gui_information();

    bool is_cli_;

    InformationContainer& info_;

    unsigned int input_fps_ = 0;

    unsigned int output_fps_ = 0;

    unsigned int saving_fps_ = 0;

    unsigned int input_throughput_ = 0;

    unsigned int output_throughput_ = 0;

    unsigned int saving_throughput_ = 0;
};
} // namespace worker
} // namespace holovibes