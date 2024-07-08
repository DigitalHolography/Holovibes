/*! \file
 *
 * \brief Implementation of the conversions between buffers.
 */
#pragma once

#include <memory>

#include <cufft.h>

#include "frame_desc.hh"
#include "batch_input_queue.hh"
#include "cuda_tools/cufft_handle.hh"
#include "function_vector.hh"
#include "enum_img_type.hh"

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
    holovibes::settings::CompositeZone,            \
    holovibes::settings::UnwrapHistorySize

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
 * \brief Manages buffer conversions for various data types and operations.
 *
 * This class encapsulates the functionalities needed to convert data between different buffer types,
 * such as Complex to Float, Float to Unsigned Short, and various other conversions required in image
 * and signal processing workflows.
 */
class Converts
{
  public:
    /*! \brief Constructor to initialize the Converts class with required settings and environments.
     *
     * \param fn_compute_vect Function vector for compute operations.
     * \param buffers Core buffer environment.
     * \param time_transformation_env Time transformation environment.
     * \param plan_unwrap_2d CUFFT handle for 2D unwrapping.
     * \param input_fd Frame descriptor for input frames.
     * \param stream CUDA stream for asynchronous operations.
     * \param settings Initialization settings.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Converts(FunctionVector& fn_compute_vect,
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

    /*! \brief Inserts functions to handle the conversion from Complex to Float.
     *
     * \param unwrap_2d_requested Indicates if 2D unwrapping is requested.
     * \param buffers_gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_float(bool unwrap_2d_requested, float* buffers_gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Float to Unsigned Short. */
    void insert_to_ushort();

    /*! \brief Inserts functions to handle the conversion from Uint(8/16/32) to Complex frame by frame.
     *
     * \param input Batch input queue containing frames to be converted.
     */
    void insert_complex_conversion(BatchInputQueue& input);

    /*! \brief Updates a specific setting.
     *
     * \tparam T Type of the setting.
     * \param setting The setting to update.
     */
    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            spdlog::trace("[Converts] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Sets pmin_ and pmax_ according to p accumulation. */
    void insert_compute_p_accu();

    /*! \brief Inserts functions to handle the conversion from Complex to Modulus.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_modulus(float* gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Complex to Squared Modulus.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_squaredmodulus(float* gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Complex to Composite.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_composite(float* gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Complex to Argument.
     *
     * \param unwrap_2d_requested Indicates if 2D unwrapping is requested.
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_argument(bool unwrap_2d_requested, float* gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Complex to Phase Increase.
     *
     * \param unwrap_2d_requested Indicates if 2D unwrapping is requested.
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_to_phase_increase(bool unwrap_2d_requested, float* gpu_postprocess_frame);

    /*! \brief Inserts functions to handle the conversion from Float to Unsigned Short in XY window. */
    void insert_main_ushort();

    /*! \brief Inserts functions to handle the conversion from Float to Unsigned Short in slices. */
    void insert_slice_ushort();

    /*! \brief Inserts functions to handle the conversion from Float to Unsigned Short for Filter2D View. */
    void insert_filter2d_ushort();

    /*! \brief Retrieves a setting value.
     *
     * \tparam T Type of the setting.
     * \return The value of the setting.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
    }

    /*! \brief Minimum p index. */
    unsigned short pmin_;

    /*! \brief Maximum value of p accumulation. */
    unsigned short pmax_;

    /*! \brief Vector of functions for compute operations. */
    FunctionVector& fn_compute_vect_;

    /*! \brief Core buffer environment. */
    const CoreBuffersEnv& buffers_;

    /*! \brief Time transformation environment. */
    const TimeTransformationEnv& time_transformation_env_;

    /*! \brief Resources for 1D phase unwrapping, used for phase increase and argument conversions. */
    std::unique_ptr<UnwrappingResources> unwrap_res_;

    /*! \brief Resources for 2D phase unwrapping, used for phase increase and argument conversions. */
    std::unique_ptr<UnwrappingResources_2d> unwrap_res_2d_;

    /*! \brief CUFFT handle for 2D unwrapping. */
    cuda_tools::CufftHandle& plan_unwrap_2d_;

    /*! \brief Descriptor for input frame size. */
    const camera::FrameDescriptor& fd_;

    /*! \brief CUDA stream for asynchronous operations. */
    const cudaStream_t& stream_;

    /*! \brief Container for real-time settings. */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
/*! \brief Checks if a setting exists in the Converts class.
 *
 * \tparam T Type of the setting.
 */
template <typename T>
struct has_setting<T, compute::Converts> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes