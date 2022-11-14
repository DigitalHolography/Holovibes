/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"
#include "import_struct.hh"

namespace holovibes
{
// clang-format off

//! \brief First frame read
class StartFrame : public UIntParameter<0, "start_frame">{};
//! \brief Last frame read
class EndFrame : public UIntParameter<0, "end_frame">{};
class ImportType : public Parameter<ImportTypeEnum, ImportTypeEnum::None, "import_type", ImportTypeEnum>{};
// FIXME : check diff with ImageType of view ; maybe the same
class LastImageType : public StringParameter<"Magnitude", "last_img_type">{};
class CurrentCameraKind : public Parameter<CameraKind, CameraKind::None, "current_camera_kind", CameraKind>{};

// clang-format on

using BasicImportCache = MicroCache<StartFrame, EndFrame, ImportType, LastImageType, CurrentCameraKind>;

// clang-format off

class ImportCache : public BasicImportCache
{
  public:
    using Base = BasicImportCache;
    class Cache : public Base::Cache{};
    class Ref : public Base::Ref{};
};

// clang-format on

} // namespace holovibes
