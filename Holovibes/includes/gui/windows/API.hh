#pragma once

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "MainWindow.hh"

namespace holovibes::api
{
std::optional<::holovibes::io_files::InputFrameFile*> import_file(const std::string& filename);

bool import_start(::holovibes::gui::MainWindow& mainwindow,
                  Holovibes& holovibes,
                  camera::FrameDescriptor& file_fd,
                  bool& is_enabled_camera,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame);

void import_stop(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes);
void close_critical_compute(::holovibes::gui::MainWindow& mainwindow, Holovibes& holovibes);

} // namespace holovibes::api