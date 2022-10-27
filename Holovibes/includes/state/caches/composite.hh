/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

#include "composite_struct.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{

using CompositeKind = CustomParameter<CompositeKindEnum, CompositeKindEnum::RGB, "composite_kind", CompositeKindEnum>;
using CompositeAutoWeights = BoolParameter<false, "composite_auto_weights">;
using CompositeRGB = CustomParameter<CompositeRGBStruct, CompositeRGBStruct{}, "composite_rgb">;
using CompositeHSV = CustomParameter<CompositeHSVStruct, CompositeHSVStruct{}, "CompositeHsv">;

using CompositeCache = MicroCache<CompositeKind, CompositeAutoWeights, CompositeRGB, CompositeHSV>;

} // namespace holovibes
