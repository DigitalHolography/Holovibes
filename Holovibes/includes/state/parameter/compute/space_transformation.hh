#pragma once

#include "custom_parameter.hh"
#include "enum_space_transformation.hh"

namespace holovibes
{

class SpaceTransformationParam : public ICustomParameter<SpaceTransformation>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = SpaceTransformation::NONE;

  public:
    SpaceTransformationParam()
        : ICustomParameter<SpaceTransformation>(DEFAULT_VALUE)
    {
    }

    SpaceTransformationParam(TransfertType value)
        : ICustomParameter<SpaceTransformation>(value)
    {
    }

  public:
    static const char* static_key() { return "space_transformation"; }
    const char* get_key() const override { return SpaceTransformationParam::static_key(); }
};

} // namespace holovibes
