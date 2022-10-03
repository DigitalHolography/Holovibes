#pragma once

#include "parameter.hh"

namespace holovibes
{
template <typename T>
class ICustomParameter : public IParameter
{
  public:
    using ValueType = T;
    using TransfertType = const T&;

  public:
    ICustomParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~ICustomParameter() override {}

  public:
    virtual TransfertType get_value() const { return value_; }
    virtual ValueType& get_value() { return value_; }
    virtual void set_value(TransfertType value) { value_ = value; }
    virtual void sync_with(IParameter* ref) override { value_ = reinterpret_cast<ICustomParameter<T>*>(ref)->value_; };

  private:
    ValueType value_;
};
} // namespace holovibes
