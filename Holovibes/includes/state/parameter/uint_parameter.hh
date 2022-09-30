#pragma once

#include "parameter.hh"

namespace holovibes
{
class IUIntParameter : public IParameter
{
  public:
    using ValueType = uint;
    using TransfertType = uint;

  public:
    IUIntParameter()
        : value_(0)
    {
    }
    IUIntParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~IUIntParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(TransfertType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<IUIntParameter*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
