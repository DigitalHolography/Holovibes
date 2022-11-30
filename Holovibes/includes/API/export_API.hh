#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline const FrameRecordStruct& get_frame_record() { return api::detail::get_value<FrameRecord>(); }
inline TriggerChangeValue<FrameRecordStruct> change_frame_record() { return api::detail::change_value<FrameRecord>(); }

inline const ChartRecordStruct& get_chart_record() { return api::detail::get_value<ChartRecord>(); }
inline TriggerChangeValue<ChartRecordStruct> change_chart_record() { return api::detail::change_value<ChartRecord>(); }

inline const std::string& get_script_path() { return api::detail::get_value<ExportScriptPath>(); }
inline void set_script_path(const std::string& value) { return api::detail::set_value<ExportScriptPath>(value); }

inline const FrameDescriptor& get_output_frame_descriptor() { return api::detail::get_value<OutputFrameDescriptor>(); }
inline TriggerChangeValue<FrameDescriptor> change_output_frame_descriptor()
{
    return api::detail::change_value<OutputFrameDescriptor>();
}

inline bool get_export_record_dont_loose_frame() { return api::detail::get_value<ExportRecordDontLoseFrame>(); }
inline void set_export_record_dont_loose_frame(bool value) { api::detail::set_value<ExportRecordDontLoseFrame>(value); }

} // namespace holovibes::api
