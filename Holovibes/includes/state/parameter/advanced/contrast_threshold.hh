#pragma once

#include "custom_parameter.hh"

namespace holovibes
{

class ContrastThresholdParam : public ICustomParameter<ContrastThreshold>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = {0.5f, 99.5f};

  public:
    ContrastThresholdParam()
        : ICustomParameter<ContrastThreshold>(DEFAULT_VALUE)
    {
    }

    ContrastThresholdParam(TransfertType value)
        : ICustomParameter<ContrastThreshold>(value)
    {
    }

  public:
    static const char* static_key() { return "contrast_threshold"; }
    const char* get_key() const override { return ContrastThresholdParam::static_key(); }
};

} // namespace holovibes
