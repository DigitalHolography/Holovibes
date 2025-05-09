/*! \file GUI.hh
 *
 * \brief This file contains the GUI API functions for the Holovibes application.
 * These functions manage UI window and recurent functions. They can call the API
 */
#pragma once

#include <memory>
#include <QMainWindow>

#include "RawWindow.hh"
#include "SliceWindow.hh"
#include "Filter2DWindow.hh"

#include "enum_computation.hh"

namespace holovibes::gui
{

/*! \brief Start the computation and open the main window
 *
 * \param window_size the size of the window
 */
void start(ushort window_size);

/*! \brief Stop the computation and close all windows */
void stop();

/*! \brief Closes all the currently displaying windows */
void close_windows();

/*! \brief Set the light ui mode
 *
 * \param value true: enable, false: disable
 */
void set_light_ui_mode(bool value);

/*! \brief Get the light ui mode
 *
 * \return Whether the light ui mode is enabled or not
 */
bool is_light_ui_mode();

/*! \brief Create and open a window of the specified size and kind
 *
 * \param[in] window_kind the kind of window to create (raw or holographic window)
 * \param[in] window_size the size of the window
 */
void create_window(Computation window_kind, ushort window_size);

/*! \brief Close all windows and reopen the current window with the new size
 *
 * \param[in] window_size the size of the window
 */
void refresh_window(ushort window_size);

/*! \brief Open or close the filter2D view
 *
 * \param[in] enabled true: open, false: close
 * \param[in] auxiliary_window_max_size the maximum size of the window
 */
void set_filter2d_view(bool enabled, uint auxiliary_window_max_size);

/*! \brief Open or close the lens view
 *
 * \param[in] enabled true: open, false: close
 * \param[in] auxiliary_window_max_size the maximum size of the window
 */
void set_lens_view(bool enabled, uint auxiliary_window_max_size);

/*! \brief Open or close the raw view
 *
 * \param[in] enabled true: open, false: close
 * \param[in] auxiliary_window_max_size the maximum size of the window
 */
void set_raw_view(bool enabled, uint auxiliary_window_max_size);

/*! \brief Open or close the plot window for the chart display
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_chart_display(bool enabled);

/*! \brief Open or close the 3D cuts view
 *
 * \param[in] enabled true: open, false: close
 * \param[in] window_size the size of the slice window
 */
void set_3d_cuts_view(bool enabled, uint max_window_size);

/*! \brief Rotates the current selected output display window (XYview or XZview or YZview) */
void rotate_texture();

/*! \brief Flips the current selected output display window (XYview or XZview or YZview) */
void flip_texture();

/*! \brief Make the ui composite overlay visible */
void set_composite_area();

/*! \brief Creates Noise overlay */
void active_noise_zone();

/*! \brief Creates Signal overlay */
void active_signal_zone();

/*! \brief Show or hide the contrast reticle overlay
 *
 * \param[in] value Whether to display the reticle overlay or not
 */
void set_contrast_reticle_overlay_visible(bool value);

/*! \brief Show or hide the reticle overlay
 *
 * \param[in] value Whether to display the reticle overlay or not
 */
void set_reticle_overlay_visible(bool value);

/*! \brief Opens additional settings window
 *
 * \param[in] parent The window.
 * \param[in] callback The function to be called on closing.
 */
void open_advanced_settings(
    QMainWindow* parent, std::function<void()> callback = []() {});

/*!
 * \brief Gets a string representation of the current recorded eye
 * This string is more explicit and should be used for display purposes.
 *
 * \return std::string The stringified recorded eye,
 * either "Left" or "Right", or "Unspecified" for no eye
 */
std::string get_recorded_eye_display_string();

/*!
 * \brief Gets a string representation of the current recorded eye
 * This string is more explicit and should be used for display purposes.
 *
 * \return std::string The stringified recorded eye,
 * either "Left" or "Right", or "Unspecified" for no eye
 */
std::string get_recorded_eye_display_string();

/*! \brief Gets the destination of the output file
 *
 * \param[in] std_filepath the output filepath.
 * \return const std::string the extension of the output file
 */
const std::string browse_record_output_file(std::string& std_filepath);

/*! \brief Returns the absolute path from a relative path (prepend by the execution directory) for qt.
 *
 * \param[in] relative_path the relative path
 * \return QString the absolute path
 */
QString create_absolute_qt_path(const std::string& relative_path);

/*! \brief Returns the absolute path to the user Documents folder.
 *
 * \return std::filesystem::path the absolute path to the user Documents folder
 */
std::filesystem::path get_user_documents_path();

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display();

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz();
std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz();

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window();
std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window();
std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window();

} // namespace holovibes::gui