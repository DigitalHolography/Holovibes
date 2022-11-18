/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"
#include "export_struct.hh"

namespace holovibes
{
// clang-format off

//! \brief Is holovibes currently recording
class FrameRecordMode : public Parameter<FrameRecordStruct, DefaultLiteral<FrameRecordStruct>{}, "frame_record_mode">{};
//! \brief Enables the signal and noise chart record
class ChartRecord : public Parameter<ChartRecordStruct, DefaultLiteral<ChartRecordStruct>{}, "chart_record">{};

// clang-format on

using ExportCache = MicroCache<FrameRecordMode, ChartRecord>;

} // namespace holovibes
