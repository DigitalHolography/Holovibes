/*! \file
 *
 * \brief Declaration of the BenchmarkWorker class.
 */
#pragma once

#include "settings/settings_container.hh"
#include "settings/settings.hh"
#include "fast_updates_types.hh"
#include "information_struct.hh"
#include "enum/enum_device.hh"
#include "worker.hh"
#include "logger.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                      \
  holovibes::settings::TimeTransformationSize

#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on
#pragma endregion

namespace holovibes::worker
{
/*! \class BenchmarkWorker
 *
 * \brief Class used to display side information relative to the execution
 */
class BenchmarkWorker final : public Worker
{
  public:
    /*!
     * \param is_cli Whether the program is running in cli mode or not
     * \param info Information container where the BenchmarkWorker periodicaly fetch data to display it
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    BenchmarkWorker(InitSettings settings)
        : Worker()
        , realtime_settings_(settings)
    {
    }

    void write_information(std::ofstream& csvFile);

    void run() override;

    static inline std::function<void(const std::string&)> display_info_text_function_;

    /*! \brief The function used to update the progress displayed */
    static inline std::function<void(ProgressType, size_t, size_t)> update_progress_function_;

    template <typename T>
    inline void update_setting(T setting)
    {
        LOG_TRACE("[BenchmarkWorker] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
        /*if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            return onrestart_settings_.get<T>().value;
        }*/
    }

    /*! \brief The map associating an indication type with its name */
    static const std::unordered_map<IndicationType, std::string> indication_type_to_string_;

    /*! \brief The map associating a fps type with its name */
    static const std::unordered_map<IntType, std::string> fps_type_to_string_;

    /*! \brief The map associating a queue type with its name */
    static const std::unordered_map<QueueType, std::string> queue_type_to_string_;

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
    void compute_throughput(size_t output_frame_res, size_t input_frame_size, size_t record_frame_size);

    /*! \brief Refresh side informations according to new computations */
    void display_gui_information();

    /*! \brief Input fps */
    size_t input_fps_ = 0;

    /*! \brief Output fps */
    size_t output_fps_ = 0;

    /*! \brief Saving fps */
    size_t saving_fps_ = 0;

    /*! \brief Camera temperature */
    size_t temperature_ = 0;

    /*! \brief Input throughput */
    size_t input_throughput_ = 0;

    /*! \brief Output throughput */
    size_t output_throughput_ = 0;

    /*! \brief Saving throughput */
    size_t saving_throughput_ = 0;

    /*! \brief Structure that will be used to retrieve information from the API */
    Information information_;

    /**
     * @brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    // DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::worker

namespace holovibes
{
template <typename T>
struct has_setting<T, worker::BenchmarkWorker> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
