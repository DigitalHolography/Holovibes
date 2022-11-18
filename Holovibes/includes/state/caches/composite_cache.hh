/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"

#include "composite_struct.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{
// clang-format off

class CompositeKind : public Parameter<CompositeKindEnum, CompositeKindEnum::RGB, "composite_kind", CompositeKindEnum>{};
class CompositeAutoWeights : public BoolParameter<false, "composite_auto_weights">{};
class CompositeRGB : public Parameter<CompositeRGBStruct, CompositeRGBStruct{}, "composite_rgb">{};
class CompositeHSV : public Parameter<CompositeHSVStruct, CompositeHSVStruct{}, "CompositeHsv">{};

// clang-format on

using CompositeCache = MicroCache<CompositeKind, CompositeAutoWeights, CompositeRGB, CompositeHSV>;

} // namespace holovibes
