/*! \file
 *  \brief Implementation of algorithms used for registration.
 *  The registration performs computations as follows:
 *  - Apply a circular mask to each images where each values outside of the circle is 0 and each
 *  values inside the circle is 1. The radius is to be chosen by the user. (hardcoded for now)
 *  - Compute the mean inside the circle for each image and substract it to rescale the data.
 *  - Apply a cross-correlation between an image choose as reference and
 *  each image of the buffer.
 *  - Take the argmax of the result of the cross-correlation, representing the point to stabilize
 *  which is the center of the eye.
 *  - Shifts all the images to stabilize them to the reference.
 *
 *  The reference is taken when the user click on the registation button in the gui.
 */
#pragma once

#include <algorithm>
#include "function_vector.hh"
#include "logger.hh"
#include "compute_env.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "masks.cuh"
#include "matrix_operations.hh"
#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                         \
    holovibes::settings::RegistrationEnabled,     \
    holovibes::settings::FftShiftEnabled,         \
    holovibes::settings::RegistrationZone

#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on

namespace holovibes
{
class Queue;
struct CoreBuffersEnv;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Registration
 *
 *  \brief Class implementation for the registration.
 *  To use the process create an object using the ctor and then just call insert_registration() in the pipe.
 */
using uint = unsigned int;

class Registration
{
  public:
    /*! \brief Constructor
     *
     *  \param[in] fn_compute_vect The vector of functions of the pipe, used to push the functions.
     *  \param[in] buffers The buffers used by the pipe, mainly used here to get `gpu_postprocess_frame`.
     *  \param[in] image_acc_env The image accumulation env, used to get `gpu_accumulation_xy_queue`.
     *  \param[in] fd The frame descriptor to get width and height.
     *  \param[in] stream The current CUDA context stream.
     *  \param[in] settings The global settings context.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Registration(FunctionVector& fn_compute_vect,
                 const CoreBuffersEnv& buffers,
                 const ImageAccEnv& image_acc_env,
                 const camera::FrameDescriptor& fd,
                 const cudaStream_t& stream,
                 InitSettings settings)
        : fn_compute_vect_(fn_compute_vect)
        , image_acc_env_(image_acc_env)
        , buffers_(buffers)
        , fd_(fd)
        , stream_(stream)
        , realtime_settings_(settings)
        , plan_2d_(fd_.width, fd_.height, CUFFT_R2C)
        , plan_2dinv_(fd_.width, fd_.height, CUFFT_C2R)
    {
        int err = !gpu_circle_mask_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_reference_image_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_current_image_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_xcorr_output_.resize(buffers_.gpu_postprocess_frame_size);
        freq_size_ = fd_.width * (fd_.height / 2 + 1);
        err += !d_freq_1_.resize(freq_size_);
        err += !d_freq_2_.resize(freq_size_);

        if (err != 0)
            throw std::exception(cudaGetErrorString(cudaGetLastError()));

        updade_cirular_mask();
    }

    /*! \brief Destructor. */
    ~Registration() = default;

    /*! \brief Insert the functions to compute the registration. The process is descripted in the head of this file.
     */
    void insert_registration();

    /*! \brief Setter for the reference image. The new reference image is taken from `buffers_.gpu_postprocess_frame`
     *  and rescaled by the mean and the `gpu_circle_mask_` is applied.
     *  This process is done so the image is ready for use in xcorr2.
     */
    void set_gpu_reference_image();

    /*! \brief Recompute the circular mask with the new radius on update. Function is also called in constructor. */
    void updade_cirular_mask();

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[Registration] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Helper function to get a settings value. */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
            return realtime_settings_.get<T>().value;
    }

    /*! \brief Preprocess the image and store it in `output` buffer.
     *  The computations are:
     *  Getting the mean and using it to rescale the `input` image while storing it in `output` buffer.
     *  Then apply the circular `gpu_circle_mask_` mask to the `output` buffer.
     *
     *  \param[out] output The output buffer used to store the processed image.
     *  \param[in] input The input buffer used to get images.
     *  \param[in out] mean Pointer to the variable to store the mean of the image being computed.
     */
    void image_preprocess(float* output, float* input, float* mean);

    /*! \brief Vector function in which we insert the processing. */
    FunctionVector& fn_compute_vect_;

    /*! \brief Main buffers used in pipe. */
    const CoreBuffersEnv& buffers_;

    /*! \brief Image Accumulation environment. This env is used to know if we have accumulated enought images before
     *  taking the reference image to compute cross-correlation.*/
    const ImageAccEnv& image_acc_env_;

    /*! \brief Describes the frame size. */
    const camera::FrameDescriptor& fd_;

    /*! \brief Compute stream to perform CUDA computations. */
    const cudaStream_t& stream_;

    /*! \brief The reference image set with the `gpu_postprocess_frame` after accumulation. Contain only one frame.
     *  This image is the one used to compute the cross-correlation with the other images to stabilize the
     *  frames.
     *  It is set each time after the images accumulation in the pipe. Hence, the reference is the accumulation of
     *  previous stabilized images. Then we have a sliding window to keep the registration in real-time.
     */
    cuda_tools::CudaUniquePtr<float> gpu_reference_image_ = nullptr;

    /*! \brief Mean of the pixels inside the cicrle of `gpu_reference_image_` after applying the mask. */
    float reference_image_mean_ = 0.0f;

    /*! \brief Float buffer containing the current `gpu_postprocess_frame` being processed. Contains only one frame.
     *  Contain the frame after applying the circular mask and rescaling. This image is used for the cross-correlation
     *  with the `gpu_reference_image_`. Also used to store the circ-shift result.
     */
    cuda_tools::CudaUniquePtr<float> gpu_current_image_ = nullptr;

    /*! \brief Mean of the pixels inside the cicrle of `gpu_current_image_` after applying the mask. */
    float current_image_mean_ = 0.0f;

    /*! \brief Buffer to store the mask after its computation. Filled with 1 inside the circle and 0 outside.
     *  The radius is to be set by the user and the mask buffer is filled in the `updade_cirular_mask`. Hence we do not
     *  call the CUDA kernel at each image but only at initialization and when the radius is modified.
     */
    cuda_tools::CudaUniquePtr<float> gpu_circle_mask_ = nullptr;

    /*! \brief Buffer to store the result of the cross-correlation, then it is used to get the argmax. */
    cuda_tools::CudaUniquePtr<float> gpu_xcorr_output_ = nullptr;

    /*! \brief The size of the buffers in the frequency domain. Used in the xcorr2 computation.
     *  It is the size required for CUFFT R2C because of Hermitian Symmetry , for more documentation refers to R2C
     *  Section of NVIDIA Cufft lib.
     */
    int freq_size_;

    /*! \brief Buffer used to store the current image after applying the R2C cufft, its size is stored in `freq_size_`.
     *  This buffer is also used to store the output image in frequency domain after applying the xcorr2 function.
     */
    cuda_tools::CudaUniquePtr<cufftComplex> d_freq_1_ = nullptr;

    /*! \brief Buffer used to store the reference image after applying the R2C cufft, its size is stored in
     * `freq_size_`.
     */
    cuda_tools::CudaUniquePtr<cufftComplex> d_freq_2_ = nullptr;

    /*! \brief Cufft plan used for R2C cufft in the xcorr2 function */
    cuda_tools::CufftHandle plan_2d_;

    /*! \brief Cufft plan used for C2R cufft in the xcorr2 function */
    cuda_tools::CufftHandle plan_2dinv_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Registration> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
