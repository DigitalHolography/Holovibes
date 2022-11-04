#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline float get_display_rate() { return api::detail::get_value<DisplayRate>(); }
inline void set_display_rate(float value) { api::detail::set_value<DisplayRate>(value); }

inline uint get_input_buffer_size() { return api::detail::get_value<InputBufferSize>(); }
inline void set_input_buffer_size(uint value) { api::detail::set_value<InputBufferSize>(value); }

inline uint get_output_buffer_size() { return api::detail::get_value<OutputBufferSize>(); }
inline void set_output_buffer_size(uint value) { api::detail::set_value<OutputBufferSize>(value); }

inline uint get_record_buffer_size() { return api::detail::get_value<RecordBufferSize>(); }
inline void set_record_buffer_size(uint value) { api::detail::set_value<RecordBufferSize>(value); }

inline uint get_file_buffer_size() { return api::detail::get_value<FileBufferSize>(); }
inline void set_file_buffer_size(uint value) { api::detail::set_value<FileBufferSize>(value); }

inline int get_raw_bitshift() { return api::detail::get_value<RawBitshift>(); }
inline void set_raw_bitshift(int value) { api::detail::set_value<RawBitshift>(value); }

inline uint get_renorm_constant() { return api::detail::get_value<RenormConstant>(); }
inline void set_renorm_constant(uint value) { api::detail::set_value<RenormConstant>(value); }

inline const Filter2DSmoothStruct& get_filter2d_smooth() { return api::detail::get_value<Filter2DSmooth>(); }
inline TriggerChangeValue<Filter2DSmoothStruct> change_filter2d_smooth()
{
    return api::detail::change_value<Filter2DSmooth>();
}

inline const ContrastThresholdStruct& get_contrast_threshold() { return api::detail::get_value<ContrastThreshold>(); }
inline TriggerChangeValue<ContrastThresholdStruct> change_contrast_threshold()
{
    return api::detail::change_value<ContrastThreshold>();
}
} // namespace holovibes::api
