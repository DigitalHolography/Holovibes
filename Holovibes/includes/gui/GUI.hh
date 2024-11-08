/*! \file
 *
 * \brief This file contains the GUI API functions for the Holovibes application. These functions manage UI window and
 * recurent functions. They can call the API
 */
#pragma once

#include <memory>

#include "RawWindow.hh"
#include "SliceWindow.hh"
#include "Filter2DWindow.hh"

namespace holovibes::gui
{

/*
void close_windows();

void init_image_mode(QPoint& position, QSize& size);

void set_raw_mode(uint window_max_size);
void set_raw_view(bool checked, uint auxiliary_window_max_size);
void create_holo_window(ushort window_size);

// Refacto set_image_mode to create the pipe and then another GUI set_image_mode
// called like (open window) that will create the windows

// View if useful since there is set_view_mode
void refresh_view_mode(ushort window_size, ImgType img_type);

void start_chart_display();

void active_noise_zone();
void active_signal_zone();

void rotateTexture();
void flipTexture();

void display_reticle(bool value);

std::string getNameFromFilename(const std::string& filename);
const std::string browse_record_output_file(std::string& std_filepath);
void record_finished(); // Record finished UID
*/

/*! \brief Open or close the filter2D view
 *
 * \param checked true: open, false: close
 * \param auxiliary_window_max_size the maximum size of the window
 */
void set_filter2d_view(bool checked, uint auxiliary_window_max_size);

/*! \brief Open or close the lens view
 *
 * \param checked true: open, false: close
 * \param auxiliary_window_max_size the maximum size of the window
 */
void set_lens_view(bool checked, uint auxiliary_window_max_size);

/*! \brief Open or close the 3D cuts view
 *
 * \param checked true: open, false: close
 * \param window_size the size of the slice window
 */
void set_3d_cuts_view(bool checked, uint window_size);

/*! \brief Make the ui composite overlay visible */
void set_composite_area();

/*! \brief Opens additional settings window
 *
 * \param parent then window that will embed the specific panel
 */
void open_advanced_settings(QMainWindow* parent = nullptr);

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz();
std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz();

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window();

} // namespace holovibes::gui