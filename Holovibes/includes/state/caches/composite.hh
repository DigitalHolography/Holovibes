/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"
#include "view_struct.hh"

namespace holovibes
{

using CompositeKindParam = CustomParameter<CompositeKind, CompositeKind::RGB, "composite_kind", CompositeKind>;
using CompositeAutoWeights = BoolParameter<false, "composite_auto_weights">;
using CompositeRGBParam = CustomParameter<CompositeRGB, CompositeRGB{}, "composite_rgb">;
using CompositeHSVParam = CustomParameter<CompositeHSV, Composite_HSV{}, "composite_hsv">;

using CompositeCache = MicroCache<CompositeKindParam, CompositeAutoWeights, CompositeRGBParam, CompositeHSVParam>;

} // namespace holovibes
