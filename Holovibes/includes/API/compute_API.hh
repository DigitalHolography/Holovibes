#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline int get_batch_size() { return api::detail::get_value<BatchSize>(); }
void set_batch_size(int value);

inline int get_time_stride() { return api::detail::get_value<TimeStride>(); }
void set_time_stride(int value);

inline const Filter2DStruct& get_filter2d() { return api::detail::get_value<Filter2D>(); }
inline TriggerChangeValue<Filter2DStruct> change_filter2d() { return api::detail::change_value<Filter2D>(); }

inline SpaceTransformationEnum get_space_transformation() { return api::detail::get_value<SpaceTransformation>(); }
inline void set_space_transformation(SpaceTransformationEnum value)
{
    api::detail::set_value<SpaceTransformation>(value);
}

inline TimeTransformationEnum get_time_transformation() { return api::detail::get_value<TimeTransformation>(); }
inline void set_time_transformation(TimeTransformationEnum value) { api::detail::set_value<TimeTransformation>(value); }

inline float get_time_transformation_size() { return api::detail::get_value<TimeTransformationSize>(); }
inline void set_time_transformation_size(float value) { api::detail::set_value<TimeTransformationSize>(value); }

inline float get_lambda() { return api::detail::get_value<Lambda>(); }
inline void set_lambda(float value) { api::detail::set_value<Lambda>(value); }

inline float get_z_distance() { return api::detail::get_value<ZDistance>(); }
inline void set_z_distance(float value) { api::detail::set_value<ZDistance>(value); }

inline ConvolutionStruct get_convolution() { return api::detail::get_value<Convolution>(); }
inline TriggerChangeValue<ConvolutionStruct> change_convolution() { return api::detail::change_value<Convolution>(); }

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
    return api::detail::get_value<TimeTransformationCutsBufferSize>();
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    api::detail::set_value<TimeTransformationCutsBufferSize>(value);
}

// other
void close_critical_compute();
void set_raw_mode(uint window_max_size);

void toggle_renormalize(bool value);

} // namespace holovibes::api
