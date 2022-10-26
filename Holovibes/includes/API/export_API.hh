#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline RecordMode get_frame_record_mode() { return api::detail::get_value<FrameRecordMode>().get_record_mode(); }
inline void set_frame_record_mode(RecordMode mode)
{
    api::detail::change_value<FrameRecordMode>()->set_record_mode(mode);
}

inline const ChartRecordStruct& get_chart_record() { return api::detail::get_value<ChartRecord>(); }
inline TriggerChangeValue<ChartRecordStruct> change_chart_record() { return api::detail::change_value<ChartRecord>(); }

} // namespace holovibes::api
