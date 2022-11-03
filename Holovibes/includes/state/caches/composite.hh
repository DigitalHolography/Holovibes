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
// clang-format off

class CompositeKind_PARAM : public CustomParameter<CompositeKindEnum, CompositeKindEnum::RGB, "composite_kind", CompositeKindEnum>{};
class CompositeAutoWeights : public BoolParameter<false, "composite_auto_weights">{};
class CompositeRGB_PARAM : public CustomParameter<CompositeRGBStruct, CompositeRGBStruct{}, "composite_rgb">{};
class CompositeHSV_PARAM : public CustomParameter<CompositeHSVStruct, CompositeHSVStruct{}, "CompositeHsv">{};

// clang-format on

class CompositeCache
    : public MicroCache<CompositeKind_PARAM, CompositeAutoWeights, CompositeRGB_PARAM, CompositeHSV_PARAM>
{
};

} // namespace holovibes
