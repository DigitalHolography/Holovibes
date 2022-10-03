#pragma once

#include "custom_parameter.hh"
#include "enum_img_type.hh"

namespace holovibes
{

class ImgTypeParam : public ICustomParameter<ImgType>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = ImgType::Modulus;

  public:
    ImgTypeParam()
        : ICustomParameter<ImgType>(DEFAULT_VALUE)
    {
    }

    ImgTypeParam(TransfertType value)
        : ICustomParameter<ImgType>(value)
    {
    }

  public:
    static const char* static_key() { return "img_type"; }
    const char* get_key() const override { return ImgTypeParam::static_key(); }
};

} // namespace holovibes
