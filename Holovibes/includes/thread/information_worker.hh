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
/*!
 *  \brief Class used to display side information relative to the execution
 */
class InformationWorker : public Worker
{
  public:
    /*!
     * \param is_cli Whether the program is running in cli mode or not
     * \param info Information container where the InformationWorker
     * periodicaly fetch data to display it
     */
    InformationWorker(bool is_cli, InformationContainer& info);

    void run() override;

  private:
    /*!
     * \brief Compute fps (input, output, saving) according to the information
     * container
     *
     * \param waited_time Time that passed since the last compute
     */
    void compute_fps(long long waited_time);

    /*!
     * \brief Compute throughput (input, output, saving) according to the
     * information container
     *
     * \param cd Compute descriptor used for computations
     * \param output_frame_res Frame resolution of output images
     * \param input_frame_size Frame size of input images
     * \param record_frame_size Frame size of record images
     */
    void compute_throughput(ComputeDescriptor& cd,
                            size_t output_frame_res,
                            size_t input_frame_size,
                            size_t record_frame_size);

    /*!
     * \brief Refresh side informations according to new computations
     */
    void display_gui_information();

    //! Whether the program is running in cli mode or not
    bool is_cli_;

    //! Information container where the InformationWorker periodicaly fetch data
    //! to display it
    InformationContainer& info_;

    //! Input fps
    size_t input_fps_ = 0;

    //! Output fps
    size_t output_fps_ = 0;

    //! Saving fps
    size_t saving_fps_ = 0;

    //! Input throughput
    size_t input_throughput_ = 0;

    //! Output throughput
    size_t output_throughput_ = 0;

    //! Saving throughput
    size_t saving_throughput_ = 0;
};
} // namespace worker
} // namespace holovibes
