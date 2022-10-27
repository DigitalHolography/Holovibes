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

//! \brief First frame read
using StartFrame = UIntParameter<0, "start_frame">;
//! \brief Last frame read
using EndFrame = UIntParameter<0, "end_frame">;
using ImportType = CustomParameter<ImportTypeEnum, ImportTypeEnum::None, "import_type", ImportTypeEnum>;
using LastImageType = StringParameter<"Magnitude", "last_img_type">;
using ConvolutionName = StringParameter<UID_CONVOLUTION_TYPE_DEFAULT, "convolution_name">;
using CurrentCameraKind = CustomParameter<CameraKind, CameraKind::None, "current_camera_kind", CameraKind>;

using ImportCache = MicroCache<StartFrame, EndFrame, ImportType, LastImageType, ConvolutionName, CurrentCameraKind>;

} // namespace holovibes
