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
/*! \brief Gets an Input file from a given filename
 *
 * \param filename the given filename to open
 * \return std::optional<io_files::InputFrameFile*> the file on success, nullopt on error
 */
std::optional<io_files::InputFrameFile*> import_file(const std::string& filename);

/*! \brief Launchs the reading of a given inputed file
 *
 * \param file_path location of the file to read
 * \param fps input fps
 * \param first_frame position of the starting frame
 * \param load_file_in_gpu if pre-loaded in gpu
 * \param last_frame position of the ending frame
 * \return true on success
 * \return false on failure
 */
bool import_start();

/*! \brief Stops the display */
void import_stop();

/**
 * \brief Sets the file start index
 */
void set_input_file_start_index(size_t value);

/**
 * \brief Sets the file end index
 */
void set_input_file_end_index(size_t value);

/*! \brief Switchs operating camera to none
 *
 */
void camera_none();

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

/*! \brief Enables the divide convolution mode
 *
 * \param value the file containing the convolution's settings
 */
void enable_convolution(const std::string& file);

void disable_convolution();

std::vector<float> get_input_filter();

/*! \brief Sets the input filter
 *
 * \param value the new value of the input filter
 */
void set_input_filter(std::vector<float> value);

/*! \brief Loads the input filter
 *
 * \param input_filter the input filter to load
 * \param file the file path
 */
void load_input_filter(std::vector<float> input_filter, const std::string& file);

/*! \brief Enables the input filter mode
 *
 * \param value the file containing the filter's settings
 */
void enable_filter();
void enable_filter(const std::string& file);

void disable_filter();

/*! \brief Sets the computation mode to Raw or Holographic*/
void set_computation_mode(Computation mode);

/*! \brief Checks preconditions to start recording
 *
 * \return success if all preconditions are met
 */
bool start_record_preconditions();

/*!
 * \brief Initiates the recording process.
 *
 * This function starts the recording process based on the current recording mode.
 * It executes the provided callback function once the recording is complete.
 *
 * \param callback A lambda function to execute at the end of the recording process.
 *                 Note: The API should not handle callbacks directly. This needs to be fixed (FIXME).
 */
void start_record(std::function<void()> callback);

/*! \brief Stops recording
 *
 * \note This functions calls the notification `record_stop` when this is done.
 */
void stop_record();

/*! \brief Change the record mode in the settings
 *
 * \param value The new record mode to be set to
 */
void set_record_mode_enum(RecordMode value);

/*!
 * \brief Gets the available extension for the given record mode
 *
 * \param mode The record mode for which to get the available extensions
 * \return std::vector<OutputFormat> The available file extensions as an enum.
 */
std::vector<OutputFormat> get_supported_formats(RecordMode mode);

/*! \brief Return whether we are recording or not
 *
 * \return true if recording, else false
 */
bool is_recording();

/*!
 * \brief Set the record queue location, between gpu and cpu
 *
 * \param gpu whether the record queue is on the gpu or the cpu
 */
void set_record_queue_location(Device device);

/*! \brief Set the record buffer size, and trigger the allocation of the pipe
 *
 * \param value the size of the buffer
 */
void set_record_buffer_size(uint value);

/*! \brief Set the camera timeout object */
void set_camera_timeout();

/*! \brief Changes the current camera used
 *
 * \param c the camera kind selection FIXME: shouldn't be stored in the wild.
 */
bool change_camera(CameraKind c);

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

/*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values
 *
 * \param composite_p_red the new value
 * \param composite_p_blue the new value
 */
void set_composite_intervals(int composite_p_red, int composite_p_blue);

/*! \brief Modifies HSV Hue min frequence
 *
 * \param composite_p_min_h the new value
 */
void set_composite_intervals_hsv_h_min(uint composite_p_min_h);

/*! \brief Modifies HSV Hue max frequence
 *
 * \param composite_p_max_h the new value
 */
void set_composite_intervals_hsv_h_max(uint composite_p_max_h);

/*! \brief Modifies HSV Hue max frequence
 *
 * \param composite_p_min_s the new value
 */
void set_composite_intervals_hsv_s_min(uint composite_p_min_s);

/*! \brief Modifies HSV Saturation max frequence
 *
 * \param composite_p_max_s the new value
 */
void set_composite_intervals_hsv_s_max(uint composite_p_max_s);

/*! \brief Modifies HSV Value min frequence
 *
 * \param composite_p_min_v the new value
 */
void set_composite_intervals_hsv_v_min(uint composite_p_min_v);

/*!
 * \brief Modifies HSV Value min frequence
 *
 * \param composite_p_max_v the new value
 */
void set_composite_intervals_hsv_v_max(uint composite_p_max_v);

/*! \brief Modifies the RGB
 *
 * \param weight_r the new value of Red
 * \param weight_g the new value of Green
 * \param weight_b the new value of blue
 */
void set_composite_weights(double weight_r, double weight_g, double weight_b);

/*! \brief Switchs between to RGB mode
 *
 */
void select_composite_rgb();

/*! \brief Switchs between to HSV mode
 *
 */
void select_composite_hsv();

/*! \brief Enables or disables Saturation frequency channel min and max
 *
 * \param composite_p_activated_s true: enable, false: disable
 */
void actualize_frequency_channel_s(bool composite_p_activated_s);

/*! \brief Enables or disables Value frequency channel min and max
 *
 * \param composite_p_activated_v true: enable, false: disable
 */
void actualize_frequency_channel_v(bool composite_p_activated_v);

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

/*! \brief Get the rounded value of max contrast for the given WindowKind
 *
 * Qt rounds the value by default.
 * In order to compare the compute descriptor values these values also needs to be rounded.
 */
float get_truncate_contrast_max(const int precision = 2);

/*! \brief Get the rounded value of min contrast for the given WindowKind
 *
 * \see get_truncate_contrast_max()
 */
float get_truncate_contrast_min(const int precision = 2);

/*! \brief Enables or Disables auto refresh contrast
 *
 * \param value true: enable, false: disable
 */
void set_contrast_auto_refresh(bool value);

/*! \brief Update the contrast of a window
 *
 * \param kind the window to update
 * \param min the min contrast value
 * \param max the max contrast value
 */
void update_contrast(WindowKind kind, float min, float max);

/*! \brief Enables or Disables log scale on the current window
 *
 * \param value true: enable, false: disable
 */
void set_log_scale(const bool value);

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
void set_reticle_zone(const units::RectFd& rect);
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

/*! \brief Gets the contrast min of a given window
 *
 * \return float the contrast minimum of the given window kind
 */
float get_contrast_min();

/*! \brief Gets the contrast max of a given window
 *
 * \return float the contrast maximum of the given window kind
 */
float get_contrast_max();

/*! \brief Gets the contrast max of a given window
 *
 * \return bool the contrast maximum of the given window kind
 */
bool get_contrast_invert();

/*! \brief Gets if the contrast is enabled for the current window
 *
 * \return bool the contrast is enabled for the current window
 */
bool get_contrast_enabled();

/*! \brief Gets the rotation of a given window
 *
 * \return double the rotation of the given window kind
 */
double get_rotation();

/*! \brief Gets the horizontal flip of a given window
 *
 * \return bool the horizontal flip of the given window kind
 */
bool get_horizontal_flip();

/*! \brief Checks if log scale is enabled for a given window
 *
 * \return true Enabled
 * \return false Disabled
 */
bool get_log_enabled();

/*! \brief Returns if the auto contrast is enabled for the current window
 *
 * \return double the rotation of the given window kind
 */
bool get_contrast_auto_refresh();

/*! \brief Disables convolution
 *
 */
void disable_convolution();

/**
 * \brief Loads a convolution matrix from a file
 *
 * This function is a tool / util supposed to be called by other functions
 *
 * \param file The name of the file to load the matrix from. NOT A FULL PATH
 * \param convo_matrix Where to store the read matrix
 *
 * \throw std::runtime_error runtime_error When the matrix cannot be loaded
 */
void load_convolution_matrix_file(const std::string& file, std::vector<float>& convo_matrix);

/*! \brief Loads convolution matrix from a given file
 *
 * \param file the file containing the convolution's settings
 */
void load_convolution_matrix(std::optional<std::string> filename);

/*! \brief Enables convolution
 *
 * \param file the file containing the convolution's settings
 */
void enable_convolution(std::optional<std::string> file);

/*! \brief Sets the contrast mode
 *
 *  \param value true: enable, false: disable
 */
void set_contrast_mode(bool contrast_enabled);

/*! \brief Sets the contrast invert mode
 *
 *  \param value true: enable, false: disable
 */
void set_contrast_invert(bool contrast_invert);

/*! \brief Sets the contrast min
 *
 *  \param value the new value
 */
void set_contrast_min(float value);

/*! \brief Sets the contrast max
 *
 *  \param value the new value
 */
void set_contrast_max(float value);

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
units::RectFd get_reticle_zone();
/*! \} */

/*! \brief Enable the divide convolution mode
 *
 * \param value true: enable, false: disable
 */
void set_divide_convolution(const bool value);

/*! \brief Creates or Removes the reticle overlay
 *
 * \param value true: create, false: remove
 */
void display_reticle(bool value);

/*! \brief Modifies reticle scale in ]0, 1[
 *
 * \param value the new reticle scale value
 */
void reticle_scale(float value);

/*! \brief Set the new value of the registration zone for the circular mask. Range ]0, 1[.
 *  \param[in] value The new zone value.
 */
void update_registration_zone(float value);

/*! \brief Start or stop the chart display
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_chart_display(bool enabled);

/*! \brief Adds or removes lens view.
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_lens_view(bool enabled);

void set_chart_display_enabled(bool value);

void set_filter2d_view_enabled(bool value);

/*! \brief Adds or removes raw view

 * \param[in] enabled true: enable, false: disable
 */
void set_raw_view(bool enabled);

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

/*! \brief Activates filter2d view */
void set_filter2d(bool checked);

/*! \brief Adds filter2d view
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_filter2d_view(bool enabled);

/*! \brief Enables or Disables renormalize image with clear image accumulation pipe
 *
 * \param value true: enable, false: disable
 */
void toggle_renormalize(bool value);

/*! \brief Enables or Disables time transform cuts views
 *
 * \param[in] enabled true: enable, false: disable
 * \return true if correctly set
 */
bool set_3d_cuts_view(bool enabled);

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

/*! \brief Configures the camera */
void configure_camera();

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

/*! \brief Update the slide value according to the bounds
 *
 * \param slider_value the new value
 * \param receiver will get the new value
 * \param bound_to_update may be updated
 * \param lower_bound the lower bound of the slide
 * \param upper_bound the upper bound of the slide
 * \return true lower_bound greater than upper_bound
 * \return false else
 */
bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound);

/*! \brief Displays information */
void start_information_display();

void* get_raw_last_image();      // get_input_queue().get()
void* get_raw_view_last_image(); // get_input_queue().get()
void* get_hologram_last_image(); // get_gpu_output_queue().get()
void* get_lens_last_image();     // api::get_compute_pipe()->get_lens_queue().get()
void* get_xz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(0).get()
void* get_yz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(1).get()
void* get_filter2d_last_image(); // api::get_compute_pipe()->get_filter2d_view_queue().get()
void* get_chart_last_image();    // api::get_compute_pipe()->get_chart_display_queue().get()

} // namespace holovibes::api

#include "API.hxx"
