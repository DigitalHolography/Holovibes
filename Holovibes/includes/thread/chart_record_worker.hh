/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

#include "settings/settings_container.hh"
#include "settings/settings.hh"

#define ONRESTART_SETTINGS               \
  holovibes::settings::RecordFilePath,   \
  holovibes::settings::RecordFrameCount 

#define ALL_SETTINGS ONRESTART_SETTINGS

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
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    ChartRecordWorker(InitSettings settings)
    : Worker()
    , onrestart_settings_(settings)
    {
      std::string file_path = setting<settings::RecordFilePath>();
      file_path = get_record_filename(file_path);
      auto nb_frames_to_record = setting<settings::RecordFrameCount>();
      onrestart_settings_.update_setting(settings::RecordFilePath{file_path});
      onrestart_settings_.update_setting(settings::RecordFrameCount{nb_frames_to_record});
    }

    void run() override;

    /**
     * @brief Update a setting. The actual application of the update
     * might ve delayed until a certain event occurs.
     * @tparam T The type of tho update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    inline void update_setting(T setting)
    {
        spdlog::info("[ChartRecordWorker] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            return onrestart_settings_.get<T>().value;
        }
    }

    /**
     * @brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::worker

namespace holovibes {
template <typename T>
struct has_setting<T, worker::ChartRecordWorker> : is_any_of<T, ALL_SETTINGS>
{
};
}