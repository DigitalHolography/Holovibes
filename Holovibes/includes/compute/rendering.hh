/*! \file
 *
 * \brief Implementation of the rendering features.
 */
#pragma once

#include <atomic>

#include "frame_desc.hh"
#include "function_vector.hh"
#include "queue.hh"
#include "rect.hh"
#include "shift_corners.cuh"
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
class ICompute;
struct CoreBuffersEnv;
struct ChartEnv;
struct TimeTransformationEnv;
struct ImageAccEnv;
} // namespace holovibes

namespace holovibes::compute
{
using uint = unsigned int;

/*! \class Rendering
 *
 * \brief #TODO Add a description for this class
 */
class Rendering
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Rendering(FunctionVector& fn_compute_vect,
              const CoreBuffersEnv& buffers,
              ChartEnv& chart_env,
              const ImageAccEnv& image_acc_env,
              const TimeTransformationEnv& time_transformation_env,
              const camera::FrameDescriptor& input_fd,
              const camera::FrameDescriptor& output_fd,
              const cudaStream_t& stream,
              ComputeCache::Cache& compute_cache,
              ExportCache::Cache& export_cache,
              ViewCache::Cache& view_cache,
              AdvancedCache::Cache& advanced_cache,
              ZoneCache::Cache& zone_cache,
              InitSettings settings)
        : fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , chart_env_(chart_env)
        , time_transformation_env_(time_transformation_env)
        , image_acc_env_(image_acc_env)
        , input_fd_(input_fd)
        , fd_(output_fd)
        , stream_(stream)
        , compute_cache_(compute_cache)
        , export_cache_(export_cache)
        , view_cache_(view_cache)
        , advanced_cache_(advanced_cache)
        , zone_cache_(zone_cache)
        , realtime_settings_(settings)
    {
        // Hold 2 float values (min and max)
        cudaXMallocHost(&percent_min_max_, 2 * sizeof(float));
    }
    ~Rendering();

    /*! \brief insert the functions relative to the fft shift. */
    void insert_fft_shift(ImgType img_type);
    /*! \brief insert the functions relative to noise and signal chart. */
    void insert_chart();
    /*! \brief insert the functions relative to the log10. */
    void insert_log();
    /*! \brief insert the functions relative to the contrast. */
    void insert_contrast(std::atomic<bool>& autocontrast_request,
                         std::atomic<bool>& autocontrast_slice_xz_request,
                         std::atomic<bool>& autocontrast_slice_yz_request,
                         std::atomic<bool>& autocontrast_filter2d_request);

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            spdlog::info("[Rendering] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief insert the log10 on the XY window */
    void insert_main_log();
    /*! \brief insert the log10 on the slices */
    void insert_slice_log();
    /*! \brief insert the log10 on the Filter2D view */
    void insert_filter2d_view_log();

    /*! \brief insert the autocontrast computation */
    void insert_compute_autocontrast(std::atomic<bool>& autocontrast_request,
                                     std::atomic<bool>& autocontrast_slice_xz_request,
                                     std::atomic<bool>& autocontrast_slice_yz_request,
                                     std::atomic<bool>& autocontrast_filter2d_request);

    /*! \brief insert the constrast on a view */
    void insert_apply_contrast(WindowKind view);

    /*! \brief Calls autocontrast and set the correct contrast variables */
    void autocontrast_caller(float* input, const uint width, const uint height, const uint offset, WindowKind view);

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
    /*! \brief Chart variables */
    ChartEnv& chart_env_;
    /*! \brief Time transformation environment */
    const TimeTransformationEnv& time_transformation_env_;
    /*! \brief Image accumulation environment */
    const ImageAccEnv& image_acc_env_;
    /*! \brief Describes the input frame size */
    const camera::FrameDescriptor& input_fd_;
    /*! \brief Describes the output frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    /*! \brief Variables needed for the computation in the pipe, updated at each end of pipe */
    ComputeCache::Cache& compute_cache_;

    ExportCache::Cache& export_cache_;
    ViewCache::Cache& view_cache_;
    AdvancedCache::Cache& advanced_cache_;
    ZoneCache::Cache& zone_cache_;

    float* percent_min_max_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Rendering> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
