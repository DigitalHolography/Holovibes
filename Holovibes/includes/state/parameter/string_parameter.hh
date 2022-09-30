#pragma once

#include "parameter.hh"

namespace holovibes
{
class IStringParameter : public IParameter
{
  public:
    using ValueType = std::string;
    using TransfertType = std::string_view;

  public:
    IStringParameter()
        : value_("")
    {
    }
    IStringParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~IStringParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(TransfertType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<IStringParameter*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
