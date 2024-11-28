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

/*! \brief Changes the time transformation size from ui value
 *
 * \param time_transformation_size The new time transformation size
 */
void update_time_transformation_size(uint time_transformation_size);

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
