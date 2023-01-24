/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

namespace holovibes::worker
{
/*! \class InformationWorker
 *
 * \brief Class used to display side information relative to the execution
 */
class InformationWorker final : public Worker
{
  public:
    /*!
     * \param is_cli Whether the program is running in cli mode or not
     * \param info Information container where the InformationWorker periodicaly fetch data to display it
     */
    InformationWorker();

    void run() override;

    static inline std::function<void(const std::string&)> display_info_text_function_;
    static inline std::function<void(ProgressType, size_t, size_t)> update_progress_function_;
    static inline std::function<void(bool)> is_input_queue_ok_;
    bool last_input_queue_state_ = false;

  private:
    /*! \brief The map associating an indication type with its name */
    static const std::unordered_map<IndicationType, std::string> indication_type_to_string_;

    /*! \brief The map associating a fps type with its name */
    static const std::unordered_map<FpsType, std::string> fps_type_to_string_;

    /*! \brief The map associating a queue type with its name */
    static const std::unordered_map<QueueType, std::string> queue_type_to_string_;

    /*! \brief Compute fps (input, output, saving) according to the information container
     *
     * \param waited_time Time that passed since the last compute
     */
    void compute_fps(long long waited_time);

    /*! \brief Compute throughput (input, output, saving) according to the information container
     *
     * \param cd Compute descriptor used for ComputeModeEnums
     * \param output_frame_res Frame resolution of output images
     * \param input_frame_size Frame size of input images
     * \param record_frame_size Frame size of record images
     */
    void compute_throughput(size_t output_frame_res, size_t input_frame_size, size_t record_frame_size);

    /*! \brief Refresh side informations according to new ComputeModeEnums */
    void display_gui_information();

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
} // namespace holovibes::worker
