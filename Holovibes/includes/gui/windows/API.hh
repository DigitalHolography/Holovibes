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
 * \param mainwindow TODO: to remove
 * \param holovibes the object used for launching
 * \param file_fd TODO: to store in an object
 * \param is_enabled_camera TODO: to store in an object
 * \param file_path
 * \param fps
 * \param first_frame
 * \param load_file_in_gpu
 * \param last_frame
 * \return true on success
 * \return false on failure
 */
bool import_start(::holovibes::gui::MainWindow& mainwindow,
                  Holovibes& holovibes,
                  camera::FrameDescriptor& file_fd,
                  bool& is_enabled_camera,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame);

/*!
 * \brief Stops the display
 *
 * \param mainwindow TODO: to remove
 * \param holovibes the object embeding the display to stop
 */
void import_stop(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes);

/*!
 * \brief Stops the program compute
 *
 * \param mainwindow TODO: to remove
 * \param holovibes the computing object
 */
void close_critical_compute(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes);

} // namespace holovibes::api