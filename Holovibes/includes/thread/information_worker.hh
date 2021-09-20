/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

namespace holovibes
{
class InformationContainer;

namespace worker
{
/*! \class InformationWorker
 *
 * \brief Class used to display side information relative to the execution
 */
class InformationWorker : public Worker
{
  public:
    /*!
     * \param is_cli Whether the program is running in cli mode or not
     * \param info Information container where the InformationWorker periodicaly fetch data to display it
     */
    InformationWorker(bool is_cli, InformationContainer& info);

    void run() override;

  private:
    /*! \brief Compute fps (input, output, saving) according to the information container
     *
     * \param waited_time Time that passed since the last compute
     */
    void compute_fps(long long waited_time);

    /*! \brief Compute throughput (input, output, saving) according to the information container
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

    /*! \brief Refresh side informations according to new computations */
    void display_gui_information();

    /*! \brief Whether the program is running in cli mode or not */
    bool is_cli_;

    /*! \brief Information container where the InformationWorker periodicaly fetch data to display it */
    InformationContainer& info_;

    /*! \brief Input fps */
    size_t input_fps_ = 0;

    /*! \brief Output fps */
    size_t output_fps_ = 0;

    /*! \brief Saving fps */
    size_t saving_fps_ = 0;

    /*! \brief Input throughput */
    size_t input_throughput_ = 0;

    /*! \brief Output throughput */
    size_t output_throughput_ = 0;

    /*! \brief Saving throughput */
    size_t saving_throughput_ = 0;
};
} // namespace worker
} // namespace holovibes
