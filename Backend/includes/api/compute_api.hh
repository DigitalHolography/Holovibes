/*! \file compute_api.hh
 *
 * \brief Regroup all functions related to computation: start/stop computation, compute mode, view mode
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

/*! \class ComputeApi
 *
 * \brief Regroup all functions related to computation: start/stop computation, compute mode, view mode and queues
 */
class ComputeApi : public IApi
{

  public:
    ComputeApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Compute

    /*! \brief Return whether the computation is stopped or not.
     *
     * \return bool True if the computation is stopped, false otherwise
     */
    inline bool get_is_computation_stopped() const { return GET_SETTING(IsComputationStopped); }

    /*! \brief Stops the pipe and the file/camera reading.
     *
     * \return ApiCode OK if the computation was stopped or NOT_STARTED if nothing needed to be stopped.
     */
    ApiCode stop() const;

    /*! \brief Starts the computation. This function will:
     * - init the input_queue if not initialized
     * - init the GPU output_queue if not initialized
     * - create a new pipe if not initialized
     * - start the computation worker
     * - start the frame read worker
     *
     * \return ApiCode OK if the computation was started or NO_IN_DATA if there is no source (file/camera).
     */
    ApiCode start() const;

#pragma endregion

    /*! \brief Returns the capacity (number of frames) of the output buffer. The output buffer stores the final frames
     * of the Holographic pipeline.
     *
     * \return size_t The current output buffer size
     */
    inline size_t get_output_buffer_size() const { return GET_SETTING(OutputBufferSize); }

    /*! \brief Sets the capacity (number of frames) of the output buffer. The output buffer stores the final frames of
     * the Holographic pipeline.
     *
     * \param[in] value The new output buffer size
     */
    inline void set_output_buffer_size(size_t value) const { UPDATE_SETTING(OutputBufferSize, value); }

    /*! \brief Return the gpu output queue.
     *
     * \return std::shared_ptr<Queue> The gpu output queue
     */
    inline std::shared_ptr<Queue> get_gpu_output_queue() const
    {
        return get_is_computation_stopped() ? nullptr : Holovibes::instance().get_compute_pipe()->get_output_queue();
    };

    /*! \brief Return the input queue.
     *
     * \return std::shared_ptr<BatchInputQueue> The input queue
     */
    inline std::shared_ptr<BatchInputQueue> get_input_queue() const
    {
        return get_is_computation_stopped() ? nullptr : Holovibes::instance().get_input_queue();
    };

#pragma endregion

#pragma region Pipe

    /*! \brief Return the compute pipe or nullptr if not initialized.
     *
     * \return std::shared_ptr<Pipe> The compute pipe
     * \throw std::runtime_error If the compute pipe is not initialized
     */
    inline std::shared_ptr<Pipe> get_compute_pipe() const
    {
        return get_is_computation_stopped() ? nullptr : Holovibes::instance().get_compute_pipe();
    };

#pragma endregion

#pragma region Img Type

    /*! \brief Returns the current view mode (Magnitude, Argument, Phase Increase, Composite Image, etc.)
     *
     * \return ImgType The current view mode
     */
    inline ImgType get_img_type() const { return GET_SETTING(ImageType); }

    /*! \brief Changes the image type (Magnitude, Argument, Phase Increase, Composite Image, etc.). If computation has
     * started requests a pipe refresh or a pipe rebuild in case of Composite.
     *
     * \param[in] type The new image type
     *
     * \return ApiCode OK if the image type was changed, NO_CHANGE if the image type was the same, WRONG_COMP_MODE if we
     * are in Raw mode or FAILURE on error.
     */
    ApiCode set_img_type(const ImgType type) const;

#pragma endregion

#pragma region Compute Mode

    /*! \brief Returns the current computation mode (Raw or Holographic)
     *
     * \return Computation The current computation mode
     */
    inline Computation get_compute_mode() const { return GET_SETTING(ComputeMode); }

    /*! \brief Sets the computation mode to Raw or Holographic. If computation has started
     *  the pipe is refreshed and the gpu output queue is rebuild.
     *
     * \param[in] mode The new computation mode.
     *
     * \return ApiCode OK if the computation mode was changed or NO_CHANGE if the computation mode was the same.
     */
    ApiCode set_compute_mode(Computation mode) const;

#pragma endregion

  private:
    /*! \brief Sets whether the computation is stopped or not.
     *
     * \param[in] value True to stop the computation, false otherwise
     */
    inline void set_is_computation_stopped(bool value) const { UPDATE_SETTING(IsComputationStopped, value); }
};

} // namespace holovibes::api