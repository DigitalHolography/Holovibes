/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"
#include "export_struct.hh"
#include "frame_desc.hh"

namespace holovibes
{
// clang-format off

class Record : public Parameter<RecordStruct, DefaultLiteral<RecordStruct>{}, "frame_record">{};
class ExportScriptPath : public StringParameter<"", "export_script_path">{};

class OutputFrameDescriptor : public Parameter<FrameDescriptor, DefaultLiteral<FrameDescriptor>{}, "output_frame_descriptor">{};

class ExportRecordDontLoseFrame: public BoolParameter<true, "export_record_dont_loose_frame">{};

// clang-format on

using ExportCache = MicroCache<Record, ExportScriptPath, OutputFrameDescriptor, ExportRecordDontLoseFrame>;

} // namespace holovibes
