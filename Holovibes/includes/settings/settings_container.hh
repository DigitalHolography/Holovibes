/**
 * @file settings_container.hh
 *
 * @brief Contains the definition of the RealtimeSettingsContainer and
 * DelayedSettingsContainer classes, as well as the has_setting helper with its
 * specializations for these two classes.
 * The RealtimeSettingsContainer is used to store settings that
 * should be updated in realtime, while the DelayedSettingsContainer
 * is used to store settings that should be updated later (eg. on restart).
 *
 * Usage:
 * - To create a new container pass a tuple of the initial values to the constructor and all the settings as template
 * parameters.
 * - To update settings in realtime, use @ref holovibes::RealtimeSettingsContainer::update_setting "update_setting".
 * - To update settings with a delay, use @ref holovibes::DelayedSettingsContainer::update_setting "update_setting"
 *   and apply delayed updates with @ref holovibes::DelayedSettingsContainer::apply_updates "apply_updates".
 * - To get the value of a setting, use @ref holovibes::SettingsContainer::get "get".
 * - To check if a setting is in a container, use @ref holovibes::has_setting "has_setting" or
 *   @ref holovibes::has_setting_v "has_setting_v".
 * Code example:
 * ```cpp
 * // Create a RealtimeSettingsContainer
 * RealtimeSettingsContainer settings<std::string, int, float>(std::make_tuple("Hello", 42, 3.14));
 *
 * // Update a setting in realtime
 * settings.update_setting(42);
 *
 * // Get the value of a setting
 * int value = settings.get<int>();
 *
 * // Check if a setting is in the container
 * has_setting_v<int, settings> // returns true
 * ```
 */

#pragma once

#include "custom_type_traits.hh"
#include <functional>
#include <spdlog/spdlog.h>
#include <tuple>
#include <type_traits>
#include "logger.hh"

namespace holovibes
{
/**
 * @brief SFINEA helper to check if a setting is in a container.
 *
 * By default it is false but it will be specialized for the
 * RealtimeSettingsContainer and the DelayedSettingsContainer.
 *
 * @tparam T The type of the setting to check.
 * @tparam SettingsContainer The container to check.
 */
template <typename T, typename SettingsContainer>
struct has_setting : std::false_type
{
};

/**
 * @brief Syntactic sugar for has_setting::value.
 *
 * @tparam T The type of the setting to check.
 * @tparam SettingsContainer The container to check.
 */
template <typename T, typename SettingsContainer>
inline constexpr bool has_setting_v = has_setting<T, SettingsContainer>::value;

template <typename... Settings>
class SettingsContainer
{
  public:
    /**
     * @brief Construct a new Settings Container object.
     *
     * Initializing a setting that is not in the container will trigger a
     * compilation error.
     *
     * @param settings The initial values of all settings.
     * @tparam InitSettings The type of the tuple used to initialize the
     * settings.
     */
    template <TupleContainsTypes<Settings...> InitSettings>
    SettingsContainer(InitSettings settings)
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
        LOG_TRACE("[SettingsContainer] [update_setting] {}", typeid(T).name());
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
 * @brief A container for settings that should be updated in realtime.
 * @tparam Settings The settings stored in the container.
 */
template <typename... Settings>
class RealtimeSettingsContainer : public SettingsContainer<Settings...>
{
  public:
    /**
     * @brief Update a setting. This specialization is for settings
     * that should be updated in realtime.
     *
     * Updating a setting that is not in the container will trigger
     * a compilation error.
     *
     * @tparam T The type of the setting to update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    enable_if_any_of<T, Settings...> inline update_setting(T setting)
    {
        LOG_TRACE("[SettingsContainer] [update_setting] {}", typeid(T).name());
        std::get<T>(this->settings_) = setting;
    }
};

/**
 * @brief SFINEA helper to check if a setting is in a container. This
 * specialization is for the RealtimeSettingsContainer.
 *
 * @tparam T The type of the setting to check.
 * @tparam ...Settings The settings stored in the container.
 */
template <typename T, typename... Settings>
struct has_setting<T, RealtimeSettingsContainer<Settings...>> : is_any_of<T, Settings...>
{
};

/**
 * @brief A container for settings that should be updated with a delay.
 * @tparam Settings The settings stored in the container.
 */
template <typename... Settings>
class DelayedSettingsContainer : public SettingsContainer<Settings...>
{
  public:
    /**
     * @brief Construct a new Settings Container object.
     *
     * Initializing a setting that is not in the container will trigger a
     * compilation error.
     *
     * @param settings The initial values of all settings.
     * @tparam InitSettings The type of the tuple used to initialize the
     * settings.
     */
    template <TupleContainsTypes<Settings...> InitSettings>
    DelayedSettingsContainer(InitSettings settings)
        : SettingsContainer<Settings...>{settings}
    {
        buffer_ = this->settings_;
    }

    /**
     * @brief Store that a setting should be updated. This specialization is for settings
     * that should be updated on restart.
     *
     * You need to call apply_updates to actually update the settings.
     * Updating a setting that is not in the container will trigger
     * a compilation error.
     *
     * @tparam T The type of the setting to update.
     * @param setting The new value of the setting.
     */
    template <typename T>
    enable_if_any_of<T, Settings...> inline update_setting(T setting)
    {
        LOG_TRACE("[SettingsContainer] [update_setting] {}", typeid(T).name());
        std::get<T>(buffer_) = setting;
    }

    /**
     * @brief Update the settings with the buffered values.
     */
    void apply_updates()
    {
        LOG_TRACE("[SettingsContainer] [apply_updates]");
        (apply_update<Settings>(), ...);
    }

  private:
    /**
     * @brief Apply the buffered update of a setting.
     * @tparam S The type of the setting to update.
     */
    template <typename S>
    void apply_update()
    {
        if (std::get<S>(buffer_) == std::get<S>(this->settings_))
            return;

        LOG_TRACE("[SettingsContainer] [apply_update] {}", typeid(S).name());
        std::get<S>(this->settings_) = std::get<S>(buffer_);
    }

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
