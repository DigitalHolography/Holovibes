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

} // namespace holovibes::api