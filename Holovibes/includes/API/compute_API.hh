#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline int get_batch_size() { return api::detail::get_value<BatchSize>(); }
void set_batch_size(int value);

inline int get_time_stride() { return api::detail::get_value<TimeStride>(); }
void set_time_stride(int value);

inline bool get_divide_convolution_enabled() { return api::detail::get_value<DivideConvolutionEnable>(); }
inline void set_divide_convolution_enabled(bool value) { api::detail::set_value<DivideConvolutionEnable>(value); }

inline float get_lambda() { return api::detail::get_value<Lambda>(); }
inline void set_lambda(float value) { api::detail::set_value<Lambda>(value); }

inline float get_time_transformation_size() { return api::detail::get_value<TimeTransformationSize>(); }
inline void set_time_transformation_size(float value) { api::detail::set_value<TimeTransformationSize>(value); }

inline SpaceTransformation get_space_transformation() { return api::detail::get_value<SpaceTransformationParam>(); }
inline void set_space_transformation(SpaceTransformation value)
{
    api::detail::set_value<SpaceTransformationParam>(value);
}

inline TimeTransformation get_time_transformation() { return api::detail::get_value<TimeTransformationParam>(); }
inline void set_time_transformation(TimeTransformation value)
{
    api::detail::set_value<TimeTransformationParam>(value);
}

inline float get_z_distance() { return api::detail::get_value<ZDistance>(); }
inline void set_z_distance(float value) { api::detail::set_value<ZDistance>(value); }

inline bool get_convolution_enabled() { return api::detail::get_value<ConvolutionEnabled>(); }
inline void set_convolution_enabled(bool value) { api::detail::set_value<ConvolutionEnabled>(value); }

inline const std::vector<float>& get_convolution_matrix() { return api::detail::get_value<ConvolutionMatrix>(); }
inline void set_convolution_matrix(const std::vector<float>& value)
{
    api::detail::set_value<ConvolutionMatrix>(value);
}

inline uint get_input_fps() { return api::detail::get_value<InputFps>(); }
inline void set_input_fps(uint value) { api::detail::set_value<InputFps>(value); }

inline Computation get_compute_mode() { return api::detail::get_value<ComputeMode>(); }
inline void set_compute_mode(Computation value) { api::detail::set_value<ComputeMode>(value); }

inline float get_pixel_size() { return api::detail::get_value<PixelSize>(); }
inline void set_pixel_size(float value) { api::detail::set_value<PixelSize>(value); }

inline uint get_unwrap_history_stopped() { return api::detail::get_value<UnwrapHistorySize>(); }
inline void set_unwrap_history_stopped(uint value) { api::detail::set_value<UnwrapHistorySize>(value); }

inline bool get_is_computation_stopped() { return api::detail::get_value<IsComputationStopped>(); }
inline void set_is_computation_stopped(bool value) { api::detail::set_value<IsComputationStopped>(value); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return api::detail::get_value<TimeTransformationCutsOutputBufferSize>();
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    api::detail::set_value<TimeTransformationCutsOutputBufferSize>(value);
}

// other
void close_critical_compute();
void set_raw_mode(uint window_max_size);

void update_batch_size(std::function<void()> notify_callback, const uint batch_size);
void update_time_stride(std::function<void()> callback, const uint time_stride);

void set_time_transformation_size(std::function<void()> callback);

void toggle_renormalize(bool value);

} // namespace holovibes::api
