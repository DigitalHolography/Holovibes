#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline const FrameRecordStruct& get_frame_record_mode() { return api::detail::get_value<FrameRecordMode>(); }
inline TriggerChangeValue<FrameRecordStruct> change_frame_record_mode()
{
    return api::detail::change_value<FrameRecordMode>();
}

inline const ChartRecordStruct& get_chart_record() { return api::detail::get_value<ChartRecord>(); }
inline TriggerChangeValue<ChartRecordStruct> change_chart_record() { return api::detail::change_value<ChartRecord>(); }

} // namespace holovibes::api
