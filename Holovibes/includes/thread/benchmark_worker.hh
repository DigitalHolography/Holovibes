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
     * \param[in] is_cli Whether the program is running in cli mode or not
     * \param[in] info Information container where the BenchmarkWorker periodically fetches data to display it
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    BenchmarkWorker(InitSettings settings)
        : Worker()
        , realtime_settings_(settings)
    {
    }

    /*!
     * \brief Writes a wave of benchmark data into the provided file
     *
     * \param[in] csvFile The file to write in (as a stream)
     */
    void write_information(std::ofstream& csvFile);

    void run() override;

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
    /*!
     * \brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
    }

    /*! \brief The map associating a queue type with its name */
    static const std::unordered_map<QueueType, std::string> queue_type_to_string_;

    /*! \brief Structure that will be used to retrieve information from the API */
    Information information_;

    /*!
     * \brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::worker

namespace holovibes
{
template <typename T>
struct has_setting<T, worker::BenchmarkWorker> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
