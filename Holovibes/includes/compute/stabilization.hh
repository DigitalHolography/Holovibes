/*! \file
 *  \brief Implementation of algorithms used for stabilization.
 *  The stabilization performs computations as follows:
 *  - Apply a circular mask to each images where each values outside of the circle is 0 and each
 *  values inside the circle is 1. The radius is to be chosen by the user. (hardcoded for now)
 *  - Compute the mean inside the circle for each image and substract it to rescale the data.
 *  - Apply a cross-correlation between an image choose as reference and
 *  each image of the buffer.
 *  - Take the argmax of the result of the cross-correlation, representing the point to stabilize
 *  which is the center of the eye.
 *  - Shifts all the images to stabilize them to the reference.
 *
 *  The reference is the mean of the last 3 images of the buffer, to have a sliding window through the
 *  time. For now we only take the first one image.
 */
#pragma once

#include <cufft.h>
#include <algorithm>
#include "function_vector.hh"
#include "logger.hh"
#include "compute_env.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "masks.cuh"
#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::StabilizationEnabled,     \
    holovibes::settings::FftShiftEnabled

#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on

namespace holovibes
{
class Queue;
struct CoreBuffersEnv;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Stabilization
 *
 * \brief Class implementation for the stabilization.
 *  To use the process create an object using the ctor and then just call insert_stabilization() in the pipe.
 */
using uint = unsigned int;

class Stabilization
{
  public:
    /*! \brief Constructor
     *  \param fn_compute_vect The
     *  \param buffers
     *  \param fd
     *  \param stream
     *  \param settings
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Stabilization(FunctionVector& fn_compute_vect,
                  const CoreBuffersEnv& buffers,
                  const camera::FrameDescriptor& fd,
                  const cudaStream_t& stream,
                  InitSettings settings)
        : fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(fd)
        , stream_(stream)
        , realtime_settings_(settings)
    {
        int err = !gpu_circle_mask_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_reference_image_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_current_image_.resize(buffers_.gpu_postprocess_frame_size);
        err += !gpu_xcorr_output_.resize(buffers_.gpu_postprocess_frame_size);
        freq_size_ = fd_.width * (fd_.height / 2 + 1); // Size required for CUFFT R2C
        cudaMalloc((void**)&(d_freq_1_), sizeof(cufftComplex) * freq_size_);
        cudaMalloc((void**)&(d_freq_2_), sizeof(cufftComplex) * freq_size_);
        cudaMalloc((void**)&(d_corr_freq_), sizeof(cufftComplex) * freq_size_);
        cufftPlan2d(&plan_2d_, fd_.width, fd_.height, CUFFT_R2C);
        cufftPlan2d(&plan_2dinv_, fd_.width, fd_.height, CUFFT_C2R);

        if (err != 0)
            throw std::exception(cudaGetErrorString(cudaGetLastError()));

        // Get the center and radius of the circle.
        float center_X = fd_.width / 2.0f;
        float center_Y = fd_.height / 2.0f;
        float radius =
            std::min(fd_.width, fd_.height) / 3.0f; // 3.0f could be change to get a different size for the circle.
        get_circular_mask(gpu_circle_mask_, center_X, center_Y, radius, fd_.width, fd_.height, stream_);
    }

    ~Stabilization()
    {
        cufftDestroy(plan_2d_);
        cufftDestroy(plan_2dinv_);
        cudaFree(d_freq_1_);
        cudaFree(d_freq_2_);
        cudaFree(d_corr_freq_);
    }

    /*! \brief insert the functions relative to the stabilization. */
    void insert_stabilization();

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[Stabilization] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
    }

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    /*! \brief Circular queue used to store the images used as reference.
     *  This queue will be used as a sliding window to keep the stabilization in real-time.
     *  The reference image will be the mean of all the images of this queue.
     */
    std::unique_ptr<Queue> reference_images_queue_ = nullptr;

    /*! \brief Number of images to store in the `reference_images_queue`. */
    uint reference_images_number_ = 3;

    /*! \brief The reference image computed from the `reference_images_queue`. Contain only one frame.
     *  This image is the one used to compute the cross-correlation with the other images to stabilize the frames.
     */
    cuda_tools::CudaUniquePtr<float> gpu_reference_image_ = nullptr;

    /*! \brief Pointer containing the mean of the pixels inside the cicrle of `gpu_reference_image` after applying the
     *  mask.
     */
    float reference_image_mean_ = 0.0f;

    /*! \brief Float buffer. Contains only one frame.
     *  Contain the frame after applying the circular mask and rescaling. This image is used for the cross-correlation
     *  with the `gpu_reference_image`.
     */
    cuda_tools::CudaUniquePtr<float> gpu_current_image_ = nullptr;

    /*! \brief Pointer containing the mean of the pixels inside the cicrle of `gpu_current_image` after applying the
     *  mask.
     */
    float current_image_mean_ = 0.0f;

    /*! \brief TODO */
    cuda_tools::CudaUniquePtr<float> gpu_circle_mask_ = nullptr;

    /*! \brief TODO */
    cuda_tools::CudaUniquePtr<float> gpu_xcorr_output_ = nullptr;

    bool ref = false;

    int freq_size_;

    cufftComplex* d_freq_1_;
    cufftComplex* d_freq_2_;
    cufftComplex* d_corr_freq_;
    cufftHandle plan_2d_;
    cufftHandle plan_2dinv_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Stabilization> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
