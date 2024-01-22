#pragma once

/**
 * @file settings_container.hh
 * @brief Contains the definition of the RealtimeSettingsContainer and
 * DelayedSettingsContainer classes, as well as the has_setting helper with its
 * specializations for these two classes. The RealtimeSettingsContainer is used
 * to store settings that should be updated in realtime, while the
 * DelayedSettingsContainer is used to store settings that should be updated
 * with a delay (e.g. on restart).
 */

#include "utils/custom_type_traits.hh"
#include <functional>
#include <spdlog/spdlog.h>
#include <tuple>
#include <type_traits>

namespace holovibes
{
/**
 * @brief SFINEA helper to check if a setting is in a container.
 * @tparam T The type of the setting to check.
 * @tparam SettingsContainer The container to check.
 */
template <typename T, typename SettingsContainer>
struct has_setting : std::false_type
{
};

/**
 * @brief A container for settings that should be updated in realtime.
 * @tparam Settings The settings stored in the container.
 */
template <typename... Settings>
class RealtimeSettingsContainer
{
  public:
    /**
     * @brief Construct a new Settings Container object.
     * @param settings The initial values of all settings.
     * @tparam InitSettings The type of the tuple used to initialize the
     * settings.
     */
    template <TupleContainsTypes<Settings...> InitSettings>
    RealtimeSettingsContainer(InitSettings settings)
    {
        auto init_setting = [&]<typename S>(S setting) { std::get<S>(settings_) = setting; };

        (init_setting(std::get<Settings>(settings)), ...);
    }

    /**
     * @brief Update a setting. This specialization is for settings
     * that should be updated in realtime.
     * @tparam T The type of the setting to update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    enable_if_any_of<T, Settings...> inline update_setting(T setting)
    {
        spdlog::trace("[SettingsContainer] [update_setting] {}", typeid(T).name());
        std::get<T>(settings_) = setting;
    }

    /**
     * @brief Get the value of a setting.
     * @tparam T The type of the setting to get.
     * @return The value of the setting.
     */
    template <typename T>
    inline T get()
    {
        return std::get<T>(settings_);
    }

  public:
    /**
     * @brief All the settings stored in the container.
     */
    std::tuple<Settings...> settings_;
};

/**
 * @brief SFINEA helper to check if a setting is in a container. This
 * specialization is for the RealtimeSettingsContainer.
 * @tparam T The type of the setting to check.
 * @tparam ...Settings The settings stored in the container.
 */
template <typename T, typename... Settings>
struct has_setting<T, RealtimeSettingsContainer<Settings...>> : is_any_of<T, Settings...>
{
};

/**
 * @brief A container for settings that should be updated with a delay.
 */
template <typename... Settings>
class DelayedSettingsContainer
{
  public:
    /**
     * @brief Construct a new Settings Container object.
     * @param all_settings The initial values of all settings.
     * @tparam InitSettings The type of the tuple used to initialize the
     * settings.
     */
    template <TupleContainsTypes<Settings...> InitSettings>
    DelayedSettingsContainer(InitSettings settings)
    {
        auto init_setting = [&]<typename S>(S setting) { std::get<S>(settings_) = setting; };

        (init_setting(std::get<Settings>(settings)), ...);
        buffer_ = settings_;
    }

    /**
     * @brief Update a setting. This specialization is for settings
     * that should be updated on restart.
     * @tparam T The type of the setting to update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    enable_if_any_of<T, Settings...> inline update_setting(T setting)
    {
        spdlog::info("[SettingsContainer] [update_setting] {}", typeid(T).name());
        std::get<T>(buffer_) = setting;
    }

    /**
     * @brief Apply the updates.
     */
    void apply_updates()
    {
        spdlog::trace("[SettingsContainer] [apply_updates]");
        (apply_update<Settings>(), ...);
    }

    /**
     * @brief Get the value of a setting.
     * @tparam T The type of the setting to get.
     * @return The value of the setting.
     */
    template <typename T>
    inline T get()
    {
        return std::get<T>(settings_);
    }

  private:
    /**
     * @brief Apply the buffered update of a setting.
     * @tparam S The type of the setting to update.
     */
    template <typename S>
    void apply_update()
    {
        if (std::get<S>(buffer_) == std::get<S>(settings_))
            return;

        spdlog::info("[SettingsContainer] [apply_update] {}", typeid(S).name());
        std::get<S>(settings_) = std::get<S>(buffer_);
    }

  public:
    /**
     * @brief All the settings stored in the container.
     */
    std::tuple<Settings...> settings_;

  private:
    /**
     * @brief The buffer used to store the updates until they are applied.
     */
    std::tuple<Settings...> buffer_;
};

/**
 * @brief SFINEA helper to check if a setting is in a container. This
 * specialization is for the DelayedSettingsContainer.
 * @tparam T The type of the setting to check.
 * @tparam ...Settings The settings stored in the container.
 */
template <typename T, typename... Settings>
struct has_setting<T, DelayedSettingsContainer<Settings...>> : is_any_of<T, Settings...>
{
};
} // namespace holovibes