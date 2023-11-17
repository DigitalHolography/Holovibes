/*! \file
 *
 * \brief Contains functions relative to image accumulation.
 */

#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "function_vector.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "rect.hh"

#include "global_state_holder.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ImageType,                \
    holovibes::settings::X,                        \
    holovibes::settings::Y,                        \
    holovibes::settings::P,                        \
    holovibes::settings::Q,                        \
    holovibes::settings::XY,                       \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::Filter2d,                 \
    holovibes::settings::CurrentWindow,            \
    holovibes::settings::LensViewEnabled,          \
    holovibes::settings::ChartDisplayEnabled,      \
    holovibes::settings::Filter2dEnabled,          \
    holovibes::settings::Filter2dViewEnabled,      \
    holovibes::settings::FftShiftEnabled,          \
    holovibes::settings::RawViewEnabled,           \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::RenormEnabled,            \
    holovibes::settings::ReticleScale
#define ALL_SETTINGS REALTIME_SETTINGS
// clang-format on

namespace holovibes
{
class Queue;
struct CoreBuffersEnv;
struct ImageAccEnv;
} // namespace holovibes

/*! \brief Contains all functions and structure for computations variables */
namespace holovibes::compute
{
/*! \class ImageAccumulation
 *
 * \brief Class that manages the image accumulation
 *
 * It manages its own buffer, initialized when needed
 * It should be a member of the Pipe class
 */
class ImageAccumulation
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    ImageAccumulation(FunctionVector& fn_compute_vect,
                      ImageAccEnv& image_acc_env,
                      const CoreBuffersEnv& buffers,
                      const camera::FrameDescriptor& fd,
                      const cudaStream_t& stream,
                      InitSettings settings)
        : fn_compute_vect_(fn_compute_vect)
        , image_acc_env_(image_acc_env)
        , buffers_(buffers)
        , fd_(fd)
        , stream_(stream)
        , realtime_settings_(settings)
    {
    }

    /*! \brief Enqueue the image accumulation.
     *
     * Should be called just after gpu_float_buffer is computed
     */
    void insert_image_accumulation(float& gpu_postprocess_frame,
                                   unsigned int& gpu_postprocess_frame_size,
                                   float& gpu_postprocess_frame_xz,
                                   float& gpu_postprocess_frame_yz);
    /*! \brief Allocate ressources for image accumulation if requested */
    void init();

    /*! \brief Free ressources for image accumulation */
    void dispose();

    /*! \brief Clear image accumulation queue */
    void clear();

    template <typename T>
    inline void update_setting(T setting)
    {
        spdlog::info("[ImageAccumulation] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Compute average on one view */
    void compute_average(std::unique_ptr<Queue>& gpu_accumulation_queue,
                         float* gpu_input_frame,
                         float* gpu_ouput_average_frame,
                         const unsigned int image_acc_level,
                         const size_t frame_res);

    /*! \brief Insert the average computation of the float frame. */
    void insert_compute_average(float& gpu_postprocess_frame,
                                unsigned int& gpu_postprocess_frame_size,
                                float& gpu_postprocess_frame_xz,
                                float& gpu_postprocess_frame_yz);

    /*! \brief Insert the copy of the corrected buffer into the float buffer. */
    void insert_copy_accumulation_result(const holovibes::ViewXYZ& const_view_xy,
                                         float* gpu_postprocess_frame,
                                         const holovibes::ViewXYZ& const_view_xz,
                                         float* gpu_postprocess_frame_xz,
                                         const holovibes::ViewXYZ& const_view_yz,
                                         float* gpu_postprocess_frame_yz);

    /*! \brief Handle the allocation of a accumulation queue and average frame */
    void allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                     cuda_tools::UniquePtr<float>& gpu_average_frame,
                                     const unsigned int accumulation_level,
                                     const camera::FrameDescriptor fd);

    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }
    }

  private:
    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Image Accumulation environment */
    ImageAccEnv& image_acc_env_;

    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes {
template <typename T>
struct has_setting<T, compute::ImageAccumulation> : is_any_of<T, ALL_SETTINGS>
{
};
}