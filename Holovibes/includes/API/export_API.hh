#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline const RecordStruct& get_record() { return api::detail::get_value<Record>(); }
inline TriggerChangeValue<RecordStruct> change_record() { return api::detail::change_value<Record>(); }

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
