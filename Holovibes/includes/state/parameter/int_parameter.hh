#pragma once

#include "parameter.hh"

namespace holovibes
{
class IIntParameter : public IParameter
{
  public:
    using ValueType = int;
    using TransfertType = int;

  public:
    IIntParameter()
        : value_(0)
    {
    }
    IIntParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~IIntParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(TransfertType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<IIntParameter*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
