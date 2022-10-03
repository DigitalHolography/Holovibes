#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class CutsContrastPOffset : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 2;

  public:
    CutsContrastPOffset()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    CutsContrastPOffset(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "cuts_contrast_p_offset"; }
    const char* get_key() const override { return CutsContrastPOffset::static_key(); }
};

} // namespace holovibes
