/*! \file
 *
 * \brief Regroup all functions related to computation: pipe (refresh, creation, ...), compute mode, view mode
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

#pragma region Settings

inline Computation get_compute_mode() { return GET_SETTING(ComputeMode); }
inline void set_compute_mode(Computation mode) { UPDATE_SETTING(ComputeMode, mode); }

inline ImgType get_img_type() { return GET_SETTING(ImageType); }
inline void set_img_type(ImgType type) { UPDATE_SETTING(ImageType, type); }

inline size_t get_output_buffer_size() { return GET_SETTING(OutputBufferSize); }
inline void set_output_buffer_size(size_t value) { UPDATE_SETTING(OutputBufferSize, value); }

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };
inline std::shared_ptr<Pipe> get_compute_pipe_no_throw() { return Holovibes::instance().get_compute_pipe_no_throw(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };
inline std::shared_ptr<BatchInputQueue> get_input_queue() { return Holovibes::instance().get_input_queue(); };

inline bool get_is_computation_stopped() { return GET_SETTING(IsComputationStopped); }
inline void set_is_computation_stopped(bool value) { UPDATE_SETTING(IsComputationStopped, value); }

#pragma endregion

/*! \brief Stops the program compute
 *
 */
void close_critical_compute();

/*! \brief Reset some values after MainWindow receives an update exception */
void handle_update_exception();

/*! \brief Stops holovibes' controllers for computation*/
void stop_all_worker_controller();

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

/*! \brief Sets the computation mode to Raw or Holographic*/
void set_computation_mode(Computation mode);

/*!
 * \brief Disables / edits numerous settings when reading a moments file
 * Most of the settings are just booleans set to false (ex: lens view)
 *
 */
void loaded_moments_data();

/*! \brief Modifies view image type
 * Changes the setting and requests a pipe refresh
 * Also requests an autocontrast refresh
 *
 * \param type The new image type
 */
ApiCode set_view_mode(const ImgType type);

} // namespace holovibes::api