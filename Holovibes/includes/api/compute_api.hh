/*! \file
 *
 * \brief Regroup all functions related to computation: pipe (refresh, creation, ...), compute mode, view mode
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

/*! \brief Return whether the computation is stopped or not.
 *
 * \return bool True if the computation is stopped, false otherwise
 */
inline bool get_is_computation_stopped() { return GET_SETTING(IsComputationStopped); }

/*! \brief Sets whether the computation is stopped or not.
 *
 * \param[in] value True to stop the computation, false otherwise
 */
inline void set_is_computation_stopped(bool value) { UPDATE_SETTING(IsComputationStopped, value); }

/*! \brief Stops the pipe. */
void close_critical_compute();

/*! \brief Reset some values after MainWindow receives an update exception */
void handle_update_exception();

/*! \brief Stops holovibes' controllers for computation*/
void stop_all_worker_controller();

/*! \brief Force batch size and time stride to be equal to 3 for moments data type. // TODO(etienne): This function
 * should not be exposed to the user.*/
void loaded_moments_data();

#pragma region Buffer

/*! \brief Returns the capacity (number of frames) of the output buffer. The output buffer stores the final frames of
 * the Holographic pipeline.
 *
 * \return size_t The current output buffer size
 */
inline size_t get_output_buffer_size() { return GET_SETTING(OutputBufferSize); }

/*! \brief Sets the capacity (number of frames) of the output buffer. The output buffer stores the final frames of the
 * Holographic pipeline.
 *
 * \param[in] value The new output buffer size
 */
inline void set_output_buffer_size(size_t value) { UPDATE_SETTING(OutputBufferSize, value); }

/*! \brief Return the gpu output queue. // TODO(etienne): This function should not be exposed to the user.
 *
 * \return std::shared_ptr<Queue> The gpu output queue
 */
inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

/*! \brief Return the input queue. // TODO(etienne): This function should not be exposed to the user.
 *
 * \return std::shared_ptr<BatchInputQueue> The input queue
 */
inline std::shared_ptr<BatchInputQueue> get_input_queue() { return Holovibes::instance().get_input_queue(); };

#pragma endregion

#pragma region Pipe

/*! \brief Return the compute pipe or throw if no pipe. // TODO(etienne): This function should not be exposed to the
 * user.
 *
 * \return std::shared_ptr<Pipe> The compute pipe
 * \throw std::runtime_error If the compute pipe is not initialized
 */
inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };

/*! \brief Return the compute pipe. // TODO(etienne): This function should not be exposed to the user.
 *
 * \return std::shared_ptr<Pipe> The compute pipe
 */
inline std::shared_ptr<Pipe> get_compute_pipe_no_throw() { return Holovibes::instance().get_compute_pipe_no_throw(); };

/*! \brief Triggers the pipe to make it refresh */
void pipe_refresh();

/*! \brief Enables the pipe refresh */
void enable_pipe_refresh();

/*! \brief Disables the pipe refresh. You must enable pipe refresh after otherwise computations will be weird. Use with
 * caution. */
void disable_pipe_refresh();

/*! \brief Creates a new pipe and start computation */
void create_pipe();

#pragma endregion

#pragma region Img Type

/*! \brief Returns the current view mode (Magnitude, Argument, Phase Increase, Composite Image, etc.)
 *
 * \return ImgType The current view mode
 */
inline ImgType get_img_type() { return GET_SETTING(ImageType); }

/*! \brief Sets the view mode (Magnitude, Argument, Phase Increase, Composite Image, etc.). It's not a realtime function
 *
 * \param[in] type The new view mode
 */
inline void set_img_type(ImgType type) { UPDATE_SETTING(ImageType, type); }

/*! \brief Changes the image type in realtime. Changes the setting and requests a pipe refresh. If the type is composite
 * the pipe is rebuild
 *
 * \param type The new image type
 */
ApiCode set_view_mode(const ImgType type);

#pragma endregion

#pragma region Compute Mode

/*! \brief Returns the current computation mode (Raw or Holographic)
 *
 * \return Computation The current computation mode
 */
inline Computation get_compute_mode() { return GET_SETTING(ComputeMode); }

/*! \brief Sets the computation mode (Raw or Holographic). It's not in realtime.
 *
 * \param[in] mode The new computation mode
 */
inline void set_compute_mode(Computation mode) { UPDATE_SETTING(ComputeMode, mode); }

/*! \brief Sets the computation mode to Raw or Holographic in realtime.
 *
 * \param[in] mode The new computation mode
 */
void set_computation_mode(Computation mode);

#pragma endregion

} // namespace holovibes::api