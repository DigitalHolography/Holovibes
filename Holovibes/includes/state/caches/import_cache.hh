/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"
#include "import_struct.hh"
#include "frame_desc.hh"

namespace holovibes
{
// clang-format off

class ImportType : public Parameter<ImportTypeEnum, ImportTypeEnum::None, "import_type", ImportTypeEnum>{};
class ImportFrameDescriptor : public Parameter<FrameDescriptor, DefaultLiteral<FrameDescriptor>{}, "import_frame_descriptor">{};

class ImportFilePath : public StringParameter<"", "import_file_path">{};
class LoadFileInGpu : public BoolParameter<true, "load_in_gpu">{};
class StartFrame : public UIntParameter<0, "start_frame">{};
class EndFrame : public UIntParameter<0, "end_frame">{};
class FileNumberOfFrame : public UIntParameter<0, "max_end_frame">{};

class LoopFile : public BoolParameter<true, "loop_file">{};
class InputFps : public UIntParameter<0, "input_fps">{};

class CurrentCameraKind : public Parameter<CameraKind, CameraKind::None, "current_camera_kind", CameraKind>{};
// clang-format on

using ImportCache = MicroCache<ImportType,
                               ImportFrameDescriptor,
                               ImportFilePath,
                               LoadFileInGpu,
                               StartFrame,
                               EndFrame,
                               FileNumberOfFrame,
                               LoopFile,
                               InputFps,
                               CurrentCameraKind>;

} // namespace holovibes
