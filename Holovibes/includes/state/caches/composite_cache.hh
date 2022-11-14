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

class CompositeKind : public Parameter<CompositeKindEnum, CompositeKindEnum::RGB, "composite_kind", CompositeKindEnum>{};
class CompositeAutoWeights : public BoolParameter<false, "composite_auto_weights">{};
class CompositeRGB : public Parameter<CompositeRGBStruct, CompositeRGBStruct{}, "composite_rgb">{};
class CompositeHSV : public Parameter<CompositeHSVStruct, CompositeHSVStruct{}, "CompositeHsv">{};

// clang-format on

using BasicCompositeCache = MicroCache<CompositeKind, CompositeAutoWeights, CompositeRGB, CompositeHSV>;

// clang-format off

class CompositeCache : public BasicCompositeCache
{
  public:
    using Base = BasicCompositeCache;
    class Cache : public Base::Cache{};
    class Ref : public Base::Ref{};
};

// clang-format on

} // namespace holovibes
