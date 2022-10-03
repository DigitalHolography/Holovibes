#pragma once

#include "parameter.hh"

namespace holovibes
{
class IBoolParameter : public IParameter
{
  public:
    using ValueType = bool;
    using TransfertType = bool;

  public:
    IBoolParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~IBoolParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(ValueType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<IBoolParameter*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
