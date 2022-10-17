/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

//! \brief Is holovibes currently recording
using FrameRecordEnable = BoolParameter<false, "frame_record_enabled">;
//! \brief Enables the signal and noise chart record
using ChartRecordEnabled = BoolParameter<false, "chart_record_enabled">;

using ExportCache = MicroCache<FrameRecordEnable, ChartRecordEnabled>;

} // namespace holovibes
