#pragma once

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "MainWindow.hh"

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
bool import_start(Holovibes& holovibes,
                  camera::FrameDescriptor& file_fd,
                  bool& is_enabled_camera,
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
void import_stop(bool& is_enabled_camera, Holovibes& holovibes);

/*! \brief Switchs operating camera to none
 *
 * \param is_enabled_camera enables or not the camera usage FIXME: shouldn't be stored in the wild.
 * \param holovibes the computing object to stop
 */
void camera_none(bool& is_enabled_camera, Holovibes& holovibes);

/*! \brief Stops the program compute
 *
 * \param holovibes the computing object
 */
void close_critical_compute(Holovibes& holovibes);

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
void set_convolution_mode(Holovibes& holovibes, const bool value);

/*!
 * \brief Removes time transformation from computation
 *
 * \param holovibes the computing object
 * \param callback FIXME: Api is not supposed to handdle callback
 */
void cancel_time_transformation_cuts(Holovibes& holovibes, std::function<void()> callback);

/*!
 * \brief Set the record frame step object
 *
 * \param record_frame_step the value to change FIXME: shouldn't be stored in the wild.
 * \param value the new value
 */
void set_record_frame_step(unsigned int& record_frame_step, int value);

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const RecordMode record_mode,
                                std::string& output_path,
                                std::string& batch_input_path);

void start_record(Holovibes& holovibes,
                  const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  const RecordMode record_mode,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback);

void stop_record(Holovibes& holovibes, const RecordMode record_mode);

const std::string browse_record_output_file(std::string& std_filepath,
                                            std::string& record_output_directory,
                                            std::string& default_output_filename);

void set_record_mode(const std::string& text, RecordMode& record_mode);

void close_windows(Holovibes& holovibes,
                   std::unique_ptr<::holovibes::gui::RawWindow>& mainDisplay,
                   std::unique_ptr<::holovibes::gui::SliceWindow>& sliceXZ,
                   std::unique_ptr<::holovibes::gui::SliceWindow>& sliceYZ,
                   std::unique_ptr<::holovibes::gui::RawWindow>& lens_window,
                   std::unique_ptr<::holovibes::gui::RawWindow>& raw_window,
                   std::unique_ptr<::holovibes::gui::Filter2DWindow>& filter2d_window,
                   std::unique_ptr<::holovibes::gui::PlotWindow>& plot_window_);

void set_computation_mode(Holovibes& holovibes, const uint image_mode_index);

void set_camera_timeout();

void change_camera(::holovibes::gui::MainWindow& mainwindow,
                   Holovibes& holovibes,
                   CameraKind c,
                   CameraKind& kCamera,
                   bool& is_enabled_camera,
                   ::holovibes::gui::MainWindow::ImportType& import_type,
                   std::unique_ptr<::holovibes::gui::RawWindow>& mainDisplay,
                   const uint image_mode_index);

void set_image_mode(::holovibes::gui::MainWindow& mainwindow,
                    Holovibes& holovibes,
                    const bool is_null_mode,
                    const uint image_mode_index);

} // namespace holovibes::api