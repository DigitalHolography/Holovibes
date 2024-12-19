/*! \file
 *
 * \brief Implmentation of the conversions between buffers.
 */
#pragma once

#include <memory>

#include <cufft.h>

#include "frame_desc.hh"
#include "batch_input_queue.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "enum_img_type.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ImageType,                \
    holovibes::settings::P,                        \
    holovibes::settings::Filter2dViewEnabled,      \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::RGB,                      \
    holovibes::settings::CompositeKind,            \
    holovibes::settings::CompositeAutoWeights,     \
    holovibes::settings::HSV,                      \
    holovibes::settings::ZFFTShift,                \
    holovibes::settings::CompositeZone

#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on

namespace holovibes
{
struct CoreBuffersEnv;
struct BatchEnv;
struct TimeTransformationEnv;
struct UnwrappingResources;
struct UnwrappingResources_2d;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Converts
 *
 * \brief class of the conversions between buffers.
 */
class Converts
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Converts(std::shared_ptr<FunctionVector> fn_compute_vect,
             const CoreBuffersEnv& buffers,
             const TimeTransformationEnv& time_transformation_env,
             cuda_tools::CufftHandle& plan_unwrap_2d,
             const camera::FrameDescriptor& input_fd,
             const cudaStream_t& stream,
             InitSettings settings)
        : pmin_(0)
        , pmax_(0)
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , time_transformation_env_(time_transformation_env)
        , plan_unwrap_2d_(plan_unwrap_2d)
        , fd_(input_fd)
        , stream_(stream)
        , realtime_settings_(settings)
    {
    }

    /*! \brief Insert functions relative to the convertion Complex => Float */
    void insert_to_float(bool unwrap_2d_requested, float* buffers_gpu_postprocess_frame);

    /*! \brief Insert functions relative to the convertion Float => Unsigned Short */
    void insert_to_ushort();

    /*! \brief Insert the conversion Uint(8/16/32) => Complex frame by frame */
    void insert_complex_conversion(BatchInputQueue& input);

    /**
     * \brief Insert a dequeue from input_queue to output.
     *
     * Note: the data manipulated should be of depth 4 (floats)
     * on both sides.
     *
     * \param input_queue[in out] The input queue to dequeue from
     * \param output[out] The buffer where to store the data.
     */
    void insert_float_dequeue(BatchInputQueue& input_queue, void* output);

    /*! \brief Insert the conversion Complex => Modulus on a batch of time transformation size frames. */
    void insert_to_modulus_moments(float* output, const ushort f_start, const ushort f_end);

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[Converts] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Set pmin_ and pmax_ according to p accumulation. */
    void insert_compute_p_accu();

    /*! \brief Insert the convertion Complex => Modulus */
    void insert_to_modulus(float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Squared Modulus */
    void insert_to_squaredmodulus(float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Composite */
    void insert_to_composite(float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Argument */
    void insert_to_argument(bool unwrap_2d_requested, float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Phase increase */
    void insert_to_phase_increase(bool unwrap_2d_requested, float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Float => Unsigned Short in XY window */
    void insert_main_ushort();

    /*! \brief Insert the convertion Float => Unsigned Short in slices. */
    void insert_slice_ushort();

    /*! \brief Insert the convertion Float => Unsigned Short of Filter2D View. */
    void insert_filter2d_ushort();

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

    /*! \brief p_index */
    unsigned short pmin_;
    /*! \brief Maximum value of p accumulation */
    unsigned short pmax_;

    /*! \brief Vector function in which we insert the processing */
    std::shared_ptr<FunctionVector> fn_compute_vect_;

    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Time transformation environment */
    const TimeTransformationEnv& time_transformation_env_;
    /*! \brief Phase unwrapping 1D. Used for phase increase and Argument. */
    std::unique_ptr<UnwrappingResources> unwrap_res_;
    /*! \brief Phase unwrapping 2D. Used for phase increase and Argument. */
    std::unique_ptr<UnwrappingResources_2d> unwrap_res_2d_;
    /*! \brief Plan 2D. Used for unwrapping. */
    cuda_tools::CufftHandle& plan_unwrap_2d_;
    /*! \brief Describes the input frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Converts> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
