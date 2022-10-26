/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"
#include "export_struct.hh"

namespace holovibes
{

//! \brief Is holovibes currently recording
using FrameRecordMode = CustomParameter<FrameRecordStruct, DefaultLiteral<FrameRecordStruct>{}, "frame_record_enabled">;
//! \brief Enables the signal and noise chart record
using ChartRecord = CustomParameter<ExportChartStruct, DefaultLiteral<ExportChartStruct>{}, "chart_record">;

using ExportCache = MicroCache<FrameRecordMode, ChartRecord>;

} // namespace holovibes
