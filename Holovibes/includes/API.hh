/*! \file
 *
 * \brief This file contains the API functions for the Holovibes application. These functions manage input files,
 * camera operations, computation settings, visualization modes, and more. The API functions are used to interact with
 * the Holovibes application from the user interface.
 */

#pragma once

#include <optional>

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "view_panel.hh"
#include "AdvancedSettingsWindow.hh"
#include "holovibes_config.hh"
#include "user_interface_descriptor.hh"
#include "compute_settings_struct.hh"
#include "enum_api_code.hh"

#include <nlohmann/json_fwd.hpp>
using json = ::nlohmann::json;

/*! \brief Return the value of setting T in the holovibes global setting
 * Usage:
 * ```cpp
 * auto value = GET_SETTING(T);
 * ```
 */
#define GET_SETTING(setting) holovibes::Holovibes::instance().get_setting<holovibes::settings::setting>().value

/*! \brief Set the value of setting T in the holovibes global setting to value
 * Usage:
 * ```cpp
 * UPDATE_SETTING(T, value);
 * ```
 */
#define UPDATE_SETTING(setting, value)                                                                                 \
    holovibes::Holovibes::instance().update_setting(holovibes::settings::setting{value})

/*! \brief Update the value.path of setting T in the holovibes global setting to value
 * Usage:
 * ```cpp
 * SET_SETTING(T, path, value);
 *
 * // Is equivalent to
 * auto t = GET_SETTING(T);
 * t.path = value;
 * UPDATE_SETTING(T, t);
 * ```
 */
#define SET_SETTING(type, path, value)                                                                                 \
    {                                                                                                                  \
        auto setting_##type = GET_SETTING(type);                                                                       \
        setting_##type.path = value;                                                                                   \
        UPDATE_SETTING(type, setting_##type);                                                                          \
    }

namespace holovibes::api
{

/*! \brief Stops the program compute
 *
 */
void close_critical_compute();

/*! \brief Reset some values after MainWindow receives an update exception */
void handle_update_exception();

/*! \brief Stops holovibes' controllers for computation*/
void stop_all_worker_controller();

/*! \brief Gets the image accumulation slice level
 *
 * \return unsigned accumulation slice level
 */
unsigned get_accumulation_level();

/*! \brief Gets the gpu input queue frame desciptor width
 *
 * \return int
 */
int get_input_queue_fd_width();

/*! \brief Gets the gpu input queue frame desciptor height object
 *
 * \return int
 */
int get_input_queue_fd_height();

/*! \brief Get the boundary of frame descriptor
 *
 * \return float boundary
 */
float get_boundary();

/*! \brief Sets the computation mode to Raw or Holographic*/
void set_computation_mode(Computation mode);

/*! \brief Triggers the pipe to make it refresh */
void pipe_refresh();

/*! \brief Enables the pipe refresh
 *
 * \param value true: enable, false: disable
 */
void enable_pipe_refresh();

/*! \brief Disables the pipe refresh. Use with caution. Usefull for mainwindow notify, which triggers numerous pipe
 * refresh.
 *
 */
void disable_pipe_refresh();

void create_pipe();

/*! \brief Modifies p accumulation
 *
 * \param p_value the new value of p accu
 */
void set_p_accu_level(uint p_value);

/*! \brief Modifies x accumulation
 *
 * \param x_value the new value of x accu
 */
void set_x_accu_level(uint x_value);

/*! \brief Modifies x cuts
 *
 * \param x_value the new value of x cuts
 */
void set_x_cuts(uint x_value);

/*! \brief Modifies y accumulation
 *
 * \param y_value the new value of y accu
 */
void set_y_accu_level(uint y_value);

/*! \brief Modifies y cuts
 *
 * \param y_value the new value of y cuts
 */
void set_y_cuts(uint y_value);

/*! \brief Modifies q accumulation
 *
 * \param is_q_accu if q accumulation is allowed
 * \param q_value the new value of q accu
 */
void set_q_accu_level(uint q_value);

/*! \brief Modifies x and y
 *
 * \param x value to modify
 * \param y value to modify
 */
void set_x_y(uint x, uint y);

/*! \brief Modifies p
 *
 * \param value the new value of p
 */
void set_p_index(uint value);

/*! \brief Modifies q
 *
 * \param value the new value of q
 */
void set_q_index(uint value);

/*! \brief Limit the value of p_index and p_acc according to time_transformation_size */
void check_p_limits();
/*! \brief Limit the value of q_index and q_acc according to time_transformation_size */
void check_q_limits();

/*! \brief Increment p by 1 */
void increment_p();

/*! \brief Decrement p by 1 */
void decrement_p();

/*!
 * \brief Modifies wave length (represented by lambda symbol in phisics)
 *
 * \param value the new value
 */
void set_lambda(float value);

/*!
 * \brief Sets the distance value for the z-coordinate.
 *
 * This function updates the internal setting for the z-coordinate distance
 * and sends a notification to the `z_distance` notifier. Additionally,
 * it refreshes the pipeline if the computation mode is not set to raw.
 *
 * \param value The new z-coordinate distance value.
 *
 * \note
 * - This function sends the notification `z_distance` with the new value when called.
 * - If the computation mode is `Computation::Raw`, the function returns immediately
 *   without updating the setting or refreshing the pipeline.
 */
void set_z_distance(float value);

/*! \brief Modifies space transform calculation
 *
 * \param value the string to match to determine the kind of space transformation
 */
void set_space_transformation(const SpaceTransformation value);

/*! \brief Modifies time transform calculation
 *
 * \param value the string to match to determine the kind of time transformation
 */
void set_time_transformation(const TimeTransformation value);

/*! \brief Enables or Disables unwrapping 2d
 *
 * \param value true: enable, false: disable
 */
void set_unwrapping_2d(const bool value);

/*! \brief Returns the current window type
 */
WindowKind get_current_window_type();

ViewWindow get_current_window();

/*! \brief Modifies the accumulation level on the current window
 *
 * \param value the new value
 */
void set_accumulation_level(int value);

/*! \brief Rotates the current selected output display window (XYview or XZview or YZview)
 *
 */
void rotateTexture();

/*! \brief Flips the current selected output display window (XYview or XZview or YZview)
 *
 */
void flipTexture();

/*! \brief Set value of raw bit shift
 *
 * \param value to set
 */
void set_raw_bitshift(unsigned int value);

/*! \name	Setter of the overlay positions.
 * \{
 */
void set_signal_zone(const units::RectFd& rect);
void set_noise_zone(const units::RectFd& rect);
void set_composite_zone(const units::RectFd& rect);
void set_zoomed_zone(const units::RectFd& rect);
/*! \} */

/*!
 * \brief Gets the raw bit shift
 *
 * \return int the raw bit shift
 */
unsigned int get_raw_bitshift();

template <typename T>
static T get_xyz_member(T xy_member, T xz_member, T yz_member)
{
    auto window = api::get_current_window_type();
    if (window == WindowKind::XYview)
        return xy_member;
    else if (window == WindowKind::XZview)
        return xz_member;
    else
        return yz_member;
}

template <typename T, typename U>
static void set_xyz_member(T xy_member, T xz_member, T yz_member, U value)
{
    auto window = api::get_current_window_type();
    if (window == WindowKind::XYview)
        xy_member(value);
    else if (window == WindowKind::XZview)
        xz_member(value);
    else
        yz_member(value);
}

template <typename T, typename U>
static void set_xyz_members(T xy_member, T xz_member, T yz_member, U value)
{
    xy_member(value);
    xz_member(value);
    yz_member(value);
}

bool get_horizontal_flip();
double get_rotation();

/**
 * \brief Helper functions to get the member of the current view
 * \tparam T is the getter function
 */
template <typename T>
static T get_view_member(T filter2d_member, T xy_member, T xz_member, T yz_member)
{
    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        return filter2d_member;
    return get_xyz_member(xy_member, xz_member, yz_member);
}

/*! \brief Sets the rotation
 *
 *  \param value the new value
 */
void set_rotation(double value);

/*! \brief Sets the horizontal flip
 *
 *  \param value the new value
 */
void set_horizontal_flip(double value);

/*! \brief get x
 *
 * \return x
 */
ViewXY get_x(void);

/*! \brief get y
 *
 * \return y
 */
ViewXY get_y(void);

/*! \brief get p
 *
 * \return p
 */
ViewPQ get_p(void);

/*! \brief get q
 *
 * \return q
 */
ViewPQ get_q(void);

/*! \name	Getter of the overlay positions.
 * \{
 */
units::RectFd get_signal_zone();
units::RectFd get_noise_zone();
units::RectFd get_composite_zone();
units::RectFd get_zoomed_zone();
/*! \} */

/*! \brief Changes the time transformation size from ui value
 *
 * \param time_transformation_size The new time transformation size
 */
void update_time_transformation_size(uint time_transformation_size);

/*! \brief Changes the focused windows
 *
 * \param index the index representing the window to select
 */
void change_window(const int index);

/*! \brief Modifies time transformation stride size from ui value
 *
 * \param time_stride the new value
 */
void update_time_stride(const uint time_stride);

/*! \brief Modifies batch size from ui value. Used when the image mode is changed ; in this case neither batch_size or
 * time_stride were modified on the GUI, so no notify is needed.
 *
 * \param batch_size the new value
 */
void update_batch_size(uint batch_size);

/*! \brief Modifies view image type
 * Changes the setting and requests a pipe refresh
 * Also requests an autocontrast refresh
 *
 * \param type The new image type
 */
ApiCode set_view_mode(const ImgType type);

/*! \brief Saves the current state of holovibes
 *
 * \param path The location of the .json file saved
 */
void save_compute_settings(const std::string& path = ::holovibes::settings::compute_settings_filepath);

json compute_settings_to_json();

/*! \brief Setups program from .json file
 *
 * \param path the path where the .json file is
 */
void load_compute_settings(const std::string& path);

/*! \brief Change buffer siwe from .json file
 *
 * \param path the path where the .json file is
 */
void import_buffer(const std::string& path);

/*! \brief Gets the documentation url
 *
 * \return const QUrl& url
 */
const QUrl get_documentation_url();

/*! \brief Gets the credits
 *
 * \return const std::vector<std::string> credits in columns
 */
constexpr std::vector<std::string> get_credits();

/*! \brief Displays information */
void start_information_display();

} // namespace holovibes::api

#include "API.hxx"
#include "composite_api.hh"
#include "record_api.hh"
#include "input_api.hh"
#include "view_api.hh"
#include "filter2d_api.hh"
#include "globalpostprocess_api.hh"
#include "windowpostprocess_api.hh"
#include "contrast_api.hh"
