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

inline float get_contrast_lower_threshold() { return api::detail::get_value<ContrastLowerThreshold>(); }
inline void set_contrast_lower_threshold(float value) { api::detail::set_value<ContrastLowerThreshold>(value); }

inline float get_contrast_upper_threshold() { return api::detail::get_value<ContrastUpperThreshold>(); }
inline void set_contrast_upper_threshold(float value) { api::detail::set_value<ContrastUpperThreshold>(value); }

inline int get_raw_bitshift() { return api::detail::get_value<RawBitshift>(); }
inline void set_raw_bitshift(int value) { api::detail::set_value<RawBitshift>(value); }

inline uint get_renorm_constant() { return api::detail::get_value<RenormConstant>(); }
inline void set_renorm_constant(uint value) { api::detail::set_value<RenormConstant>(value); }

inline uint get_cuts_contrast_p_offset() { return api::detail::get_value<CutsContrastPOffset>(); }
inline void set_cuts_contrast_p_offset(uint value) { api::detail::set_value<CutsContrastPOffset>(value); }

} // namespace holovibes::api
