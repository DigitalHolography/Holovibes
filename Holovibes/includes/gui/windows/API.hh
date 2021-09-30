#pragma once

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "MainWindow.hh"
#include "user_interface_descriptor.hh"

namespace holovibes::api
{
/*! \brief Gets an Input file from a given filename
 *
 * \param filename the given filename to open
 * \return std::optional<::holovibes::io_files::InputFrameFile*> the file on success, nullopt on error
 */
std::optional<::holovibes::io_files::InputFrameFile*> import_file(const std::string& filename);

/*!
 * \brief Launchs the reading of a given inputed file
 *
 * \param holovibes the object used for launching
 * \param file_fd FIXME: shouldn't be stored in the wild.
 * \param is_enabled_camera FIXME: shouldn't be stored in the wild.
 * \param file_path
 * \param fps
 * \param first_frame
 * \param load_file_in_gpu
 * \param last_frame
 * \return true on success
 * \return false on failure
 */
bool import_start(UserInterfaceDescriptor& ui_descriptor,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame);

/*! \brief Stops the display
 *
 * \param is_enabled_camera enables or not the camera usage FIXME: shouldn't be stored in the wild.
 * \param holovibes the object embeding the display to stop
 */
void import_stop(UserInterfaceDescriptor& ui_descriptor);

/*! \brief Switchs operating camera to none
 *
 * \param is_enabled_camera enables or not the camera usage FIXME: shouldn't be stored in the wild.
 * \param holovibes the computing object to stop
 */
void camera_none(UserInterfaceDescriptor& ui_descriptor);

/*! \brief Stops the program compute
 *
 * \param holovibes the computing object
 */
void close_critical_compute(UserInterfaceDescriptor& ui_descriptor);

/*! \brief Removes info container in holovibes */
void remove_infos();

/*! \brief Checks if we are currently in raw mode
 *
 * \param holovibes the computing object
 * \return true if we are in raw mode, false otherwise
 */
bool is_raw_mode(Holovibes& holovibes);

/*! \brief Enables the divide convolution mode
 *
 * \param holovibes the computing object
 * \param value true: enable, false: disable
 */
void set_convolution_mode(UserInterfaceDescriptor& ui_descriptor, const bool value);

/*!
 * \brief Removes time transformation from computation
 *
 * \param holovibes the computing object
 * \param callback FIXME: Api is not supposed to handdle callback
 */
void cancel_time_transformation_cuts(UserInterfaceDescriptor& ui_descriptor, std::function<void()> callback);

/*!
 * \brief Set the record frame step object
 *
 * \param record_frame_step the value to change FIXME: shouldn't be stored in the wild.
 * \param value the new value
 */
void set_record_frame_step(UserInterfaceDescriptor& ui_descriptor, int value);

/*! \brief Checks preconditions to start recording
 *
 * \param batch_enabled if batch recording is enabled FIXME: shouldn't be stored in the wild.
 * \param nb_frame_checked if number of frame is allowed FIXME: shouldn't be stored in the wild.
 * \param nb_frames_to_record number of frame to record FIXME: shouldn't be stored in the wild.
 * \param record_mode what kind of record is it FIXME: shouldn't be stored in the wild.
 * \param output_path where to locate the destination file FIXME: shouldn't be stored in the wild.
 * \param batch_input_path where is located the input batch file FIXME: shouldn't be stored in the wild.
 * \return true on success
 * \return false on failure
 */
bool start_record_preconditions(const UserInterfaceDescriptor& ui_descriptor,
                                const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path);

/*! \brief Launchs recording
 *
 * \param holovibes the computing object
 * \param batch_enabled if batch recording is enabled FIXME: shouldn't be stored in the wild.
 * \param nb_frames_to_record number of frame to record FIXME: shouldn't be stored in the wild.
 * \param record_mode what kind of record is it FIXME: shouldn't be stored in the wild.
 * \param output_path where to locate the destination file FIXME: shouldn't be stored in the wild.
 * \param batch_input_path where is located the input batch file FIXME: shouldn't be stored in the wild.
 * \param callback lambda to execute at the end of the processing FIXME: Api is not supposed to handdle callback
 */
void start_record(UserInterfaceDescriptor& ui_descriptor,
                  const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback);

/*! \brief Stops recording
 *
 * \param holovibes the computing object
 * \param record_mode what kind of record is it FIXME: shouldn't be stored in the wild.
 */
void stop_record(UserInterfaceDescriptor& ui_descriptor);

/*! \brief Gets the destination of the output file
 *
 * \param std_filepath the output filepath FIXME: shouldn't be stored in the wild.
 * \param record_output_directory output directory FIXME: shouldn't be stored in the wild.
 * \param default_output_filename output filename FIXME: shouldn't be stored in the wild.
 * \return const std::string the extension of the output file
 */
const std::string browse_record_output_file(std::string& std_filepath,
                                            std::string& record_output_directory,
                                            std::string& default_output_filename);

/*! \brief Set the record mode object
 *
 * \param text the catched mode
 * \param record_mode record mode to modify FIXME: shouldn't be stored in the wild.
 */
void set_record_mode(UserInterfaceDescriptor& ui_descriptor, const std::string& text);

/*! \brief Closes all the currently displaying windows
 *
 * \param holovibes the computing object
 * \param mainDisplay a window FIXME: shouldn't be stored in the wild.
 * \param sliceXZ a window FIXME: shouldn't be stored in the wild.
 * \param sliceYZ a window FIXME: shouldn't be stored in the wild.
 * \param lens_window a window FIXME: shouldn't be stored in the wild.
 * \param raw_window a window FIXME: shouldn't be stored in the wild.
 * \param filter2d_window a window FIXME: shouldn't be stored in the wild.
 * \param plot_window_ a window FIXME: shouldn't be stored in the wild.
 */
void close_windows(UserInterfaceDescriptor& ui_descriptor);

/*! \brief Sets the computation mode
 *
 * \param holovibes the computing object
 * \param image_mode_index the image mode corresponding to the selection FIXME: shouldn't be stored in the wild.
 */
void set_computation_mode(Holovibes& holovibes, const uint image_mode_index);

/*! \brief Set the camera timeout object */
void set_camera_timeout();

/*! \brief Changes the current camera used
 *
 * \param mainwindow FIXME: bridge between API and MainWindow before MainWindow's methods moved to API
 * \param holovibes the computing object
 * \param c the camera kind selection FIXME: shouldn't be stored in the wild.
 * \param kCamera the current camera kind to modify FIXME: shouldn't be stored in the wild.
 * \param is_enabled_camera if camera is enablde FIXME: shouldn't be stored in the wild.
 * \param import_type the kind of import FIXME: shouldn't be stored in the wild.
 * \param mainDisplay a window FIXME: shouldn't be stored in the wild.
 * \param image_mode_index the image mode corresponding to the selection FIXME: shouldn't be stored in the wild.
 */
void change_camera(::holovibes::gui::MainWindow& mainwindow,
                   UserInterfaceDescriptor& ui_descriptor,
                   CameraKind c,
                   const uint image_mode_index);

/*! \brief Sets the image mode
 *
 * \param mainwindow a window
 * \param holovibes the computing object
 * \param is_null_mode if the selection is null
 * \param image_mode_index the image mode corresponding to the selection FIXME: shouldn't be stored in the wild.
 */
void set_image_mode(::holovibes::gui::MainWindow& mainwindow,
                    UserInterfaceDescriptor& ui_descriptor,
                    const bool is_null_mode,
                    const uint image_mode_index);

} // namespace holovibes::api