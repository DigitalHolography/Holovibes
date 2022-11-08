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
// clang-format off

//! \brief Is holovibes currently recording
class FrameRecordMode : public CustomParameter<FrameRecordStruct, DefaultLiteral<FrameRecordStruct>{}, "frame_record_enabled">{};
//! \brief Enables the signal and noise chart record
class ChartRecord : public CustomParameter<ChartRecordStruct, DefaultLiteral<ChartRecordStruct>{}, "chart_record">{};

// clang-format on

using BasicExportCache = MicroCache<FrameRecordMode, ChartRecord>;

// clang-format off

class ExportCache : public BasicExportCache
{
  public:
    using Base = BasicExportCache;
    class Cache : public Base::Cache{};
    class Ref : public Base::Ref{};
};

// clang-format on

} // namespace holovibes
