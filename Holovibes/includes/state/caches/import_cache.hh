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

//! \brief First frame read
class StartFrame : public UIntParameter<0, "start_frame">{};
//! \brief Last frame read
class EndFrame : public UIntParameter<0, "end_frame">{};

class FileNumberOfFrame : public UIntParameter<0, "max_end_frame">{};
class ImportFrameDescriptor : public Parameter<camera::FrameDescriptor, DefaultLiteral<camera::FrameDescriptor>{}, "import_frame_descriptor">{};

class ImportType : public Parameter<ImportTypeEnum, ImportTypeEnum::None, "import_type", ImportTypeEnum>{};
// FIXME : check diff with ImageType of view ; maybe the same
class LastImageType : public StringParameter<"Magnitude", "last_img_type">{};
class CurrentCameraKind : public Parameter<CameraKind, CameraKind::None, "current_camera_kind", CameraKind>{};

// clang-format on

using ImportCache = MicroCache<StartFrame,
                               EndFrame,
                               ImportType,
                               LastImageType,
                               CurrentCameraKind,
                               FileNumberOfFrame,
                               ImportFrameDescriptor>;

} // namespace holovibes
