#pragma once

#include "parameter.hh"

namespace holovibes
{
class IFloatParameter : public IParameter
{
  public:
    using ValueType = float;
    using TransfertType = float;

  public:
    IFloatParameter()
        : value_(0)
    {
    }
    IFloatParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~IFloatParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(TransfertType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<IFloatParameter*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
