/*! \file
 *
 * \brief #TODO Add a description for this file
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
#include "global_state_holder.hh"
#include "compute_settings_struct.hh"

#include <nlohmann/json_fwd.hpp>
using json = ::nlohmann::json;

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
bool import_start(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame);

/*! \brief Stops the display */
void import_stop();

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
unsigned get_img_accu_level();

/*! \brief Gets the gpu input queue frame desciptor width
 *
 * \return int
 */
int get_gpu_input_queue_fd_width();

/*! \brief Gets the gpu input queue frame desciptor height object
 *
 * \return int
 */
int get_gpu_input_queue_fd_height();

/*! \brief Get the boundary of frame descriptor
 *
 * \return float boundary
 */
float get_boundary();

/*! \brief Checks if we have an input queue
 *
 * \return true on success
 * \return false on failure
 */
bool is_gpu_input_queue();

/*! \brief Enables the divide convolution mode
 *
 * \param value the file containing the convolution's settings
 */
void enable_convolution(const std::string& file);

void disable_convolution();

/*! \brief Enables the input filter mode
 *
 * \param value the file containing the filter's settings
 */
void enable_filter(const std::string& file);

void disable_filter();

/*! \brief Changes display mode to Raw */
void set_raw_mode(uint window_max_size);

/*! \brief Changes display mode to Holographic
 *
 * \param window_size the size of the window
 * \return true on success
 * \return false on failure
 */
bool set_holographic_mode(ushort window_size);

/*! \brief Restarts everything to change the view mode
 *
 * \param window_size the size of the window
 * \param index the index on the new mode
 */
void refresh_view_mode(ushort window_size, uint index);

/*! \brief Removes time transformation from computation
 *
 * \param callback FIXME: Api is not supposed to handdle callback
 */
void cancel_time_transformation_cuts(std::function<void()> callback);

/*! \brief Checks preconditions to start recording
 *
 * \param batch_enabled if batch recording is enabled FIXME: shouldn't be stored in the wild.
 * \param nb_frame_checked if number of frame is allowed FIXME: shouldn't be stored in the wild.
 * \param nb_frames_to_record number of frame to record FIXME: shouldn't be stored in the wild.
 * \param batch_input_path where is located the input batch file FIXME: shouldn't be stored in the wild.
 * \return true on success
 * \return false on failure
 */
bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path);

/*! \brief Launchs recording
 *
 * \param batch_enabled if batch recording is enabled FIXME: shouldn't be stored in the wild.
 * \param nb_frames_to_record number of frame to record FIXME: shouldn't be stored in the wild.
 * \param output_path where to locate the destination file FIXME: shouldn't be stored in the wild.
 * \param batch_input_path where is located the input batch file FIXME: shouldn't be stored in the wild.
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handdle callback
 */
void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback);

void set_record_device(const bool gpu);

/*! \brief Stops recording
 *
 */
void stop_record();

/*! \brief Gets the destination of the output file
 *
 * \param std_filepath the output filepath FIXME: shouldn't be stored in the wild.
 * \return const std::string the extension of the output file
 */
const std::string browse_record_output_file(std::string& std_filepath);

/*! \brief Set the record mode object, and trigger the allocation of the pipe
 *
 * \param text the catched mode
 * \param record_mode record mode to modify FIXME: shouldn't be stored in the wild.
 */
void set_record_mode(const std::string& text);

/*!
 * \brief Set the record queue location, between gpu and cpu
 * 
 * \param gpu whether the record queue is on the gpu or the cpu
 */
void set_record_queue_location(bool gpu);

/*! \brief Set the record buffer size, and trigger the allocation of the pipe
 *
 * \param value the size of the buffer
 */
void set_record_buffer_size(uint value);

/*! \brief Closes all the currently displaying windows
 *
 */
void close_windows();

/*! \brief Set the camera timeout object */
void set_camera_timeout();

/*! \brief Changes the current camera used
 *
 * \param c the camera kind selection FIXME: shouldn't be stored in the wild.
 */
bool change_camera(CameraKind c);

/*! \brief Triggers the pipe to make it refresh */
void pipe_refresh();
void create_holo_window(ushort window_size);
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

/*! \brief Enables or disables Hue gaussian blur
 *
 * \param h_blur_activated true: enable, false: disable
 */
void actualize_selection_h_gaussian_blur(bool h_blur_activated);

/*! \brief Limit the value of p_index and p_acc according to time_transformation_size */
void check_p_limits();
/*! \brief Limit the value of q_index and q_acc according to time_transformation_size */
void check_q_limits();

/*! \brief Modified Hue blur size
 *
 * \param h_blur_kernel_size the new value
 */
void actualize_kernel_size_blur(uint h_blur_kernel_size);

/*! \brief Increment p by 1 */
void increment_p();

/*! \brief Decrement p by 1 */
void decrement_p();

/*!
 * \brief Modifies wave length (represented by lambda symbol in phisics)
 *
 * \param value the new value
 */
void set_wavelength(double value);

/*! \brief Modifies z
 *
 * \param value the new value
 */
void set_z_distance(const double value);

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

/*! \brief Modifies the accumulation level on the current window
 *
 * \param value the new value
 */
void set_accumulation_level(int value);

/*! \brief Make the ui compisite overlay visible
 *
 */
void set_composite_area();

/*! \brief Rotates the current selected output display window (XYview or XZview or YZview)
 *
 */
void rotateTexture();

/*! \brief Flips the current selected output display window (XYview or XZview or YZview)
 *
 */
void flipTexture();

/*! \brief Enables or Disables the contrast mode and update the current focused window
 *
 * \param value true: enable, false: disable
 */
void set_contrast_mode(bool value);

/*! \brief Adds auto contrast to the pipe over cut views
 *
 */
void set_auto_contrast_cuts();

/*! \brief Adds auto contrast to the current window
 *
 * \return true on success
 * \return false on failure
 */
bool set_auto_contrast();

/*! \brief Set the auto contrast to all windows */
void set_auto_contrast_all();

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

/*! \brief Modifies the min contrast value on the current window
 *
 * \param value the new value
 */
void set_contrast_min(const double value);

/*! \brief Modifies the max contrast value on the current window
 *
 * \param value the new value
 */
void set_contrast_max(const double value);

/*! \brief Enables or Disables contrast invertion
 *
 * \param value true: enable, false: disable
 */
void invert_contrast(bool value);

/*! \brief Enables or Disables auto refresh contrast
 *
 * \param value true: enable, false: disable
 */
void set_auto_refresh_contrast(bool value);

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

/*!
 * \brief Gets the contrast min of a given window
 *
 * \return float the contrast minimum of the given window kind
 */
float get_contrast_min();

/*!
 * \brief Gets the contrast max of a given window
 *
 * \return float the contrast maximum of the given window kind
 */
float get_contrast_max();

/*!
 * \brief Gets the contrast max of a given window
 *
 * \return bool the contrast maximum of the given window kind
 */
bool get_contrast_invert_enabled();

/*!
 * \brief Checks if log scale is enabled for a given window
 *
 * \return true Enabled
 * \return false Disabled
 */
bool get_img_log_scale_slice_enabled();

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

/*! \brief Restores attributs when recording ends
 *
 */
void record_finished();

/*! \brief Creates Noise overlay
 *
 */
void active_noise_zone();

/*! \brief Creates Signal overlay
 *
 */
void active_signal_zone();

/*! \brief Opens Chart window
 *
 */
void start_chart_display();

/*! \brief Closes Chart window
 *
 */
void stop_chart_display();

/*! \brief Adds or removes lens view */
void set_lens_view(bool checked, uint auxiliary_window_max_size);

/*! \brief Adds or removes raw view */
void set_raw_view(bool checked, uint auxiliary_window_max_size);

/*! \brief Changes the time transformation size from ui value
 *
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handle callback
 */
void set_time_transformation_size(std::function<void()> callback);

/*! \brief Changes the focused windows
 *
 * \param index the index representing the window to select
 */
void change_window(const int index);

/*! \brief Activates filter2d view */
void set_filter2d(bool checked);

/*! \brief Adds filter2d view
 *
 * \param auxiliary_window_max_size
 */
void set_filter2d_view(bool check, uint auxiliary_window_max_size);

/*! \brief Enables or Disables renormalize image with clear image accumulation pipe
 *
 * \param value true: enable, false: disable
 */
void toggle_renormalize(bool value);

/*! \brief Enables or Disables time transform cuts views
 *
 * \return true on success
 * \return false on failure
 */
bool set_3d_cuts_view(uint time_transformation_size);

/*! \brief Modifies time transformation stride size from ui value
 *
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handle callback
 * \param time_stride the new value
 */
void update_time_stride(std::function<void()> callback, const uint time_stride);

/*! \brief Modifies batch size from ui value
 *
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handle callback
 * \param batch_size the new value
 */
void update_batch_size(std::function<void()> callback, const uint batch_size);

/*! \brief Modifies view image type
 *
 * \param value The new image type
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handle callback
 */
void set_view_mode(const std::string& value, std::function<void()> callback);

/*! \brief Restarts everything to change the view mode
 *
 */

/*! \brief Changes display mode to Holographic
 *
 * \param window_size size of the window
 * \param fd the frame descriptor that will be initialized and returned by reference
 * \return true on success
 * \return false on failure
 */

/*! \brief Creates the windows for processed image output
 *
 * \param window_size the size of the window
 */

/*! \brief Creates the pipeline
 *
 */

/*!
 * \brief Set the raw mode object
 *
 */

/*! \brief Configures the camera */
void configure_camera();

/*! \brief Gets data from the current main display
 *
 * \param position the position to fill
 * \param size the size to fill
 */
void init_image_mode(QPoint& position, QSize& size);

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

/*! \brief Gets the documentation url
 *
 * \return const QUrl& url
 */
const QUrl get_documentation_url();

/*! \brief Gets the credits
 *
 * \return const std::string credits
 */
const std::string get_credits();

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

/*! \brief Displays information
 *
 * \param is_cli true if the current user interface is CLI
 * \param callback lambda to execute FIXME: Api is not supposed to handdle callback
 */
void start_information_display(const std::function<void()>& callback = []() {});

/*! \brief Opens additional settings window
 *
 * \param parent then window that will embed the specific panel
 * \param specific_panel the specific panel to link
 */
void open_advanced_settings(QMainWindow* parent = nullptr,
                            ::holovibes::gui::AdvancedSettingsWindowPanel* specific_panel = nullptr);

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz();
std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz();

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window();

} // namespace holovibes::api

#include "API.hxx"
