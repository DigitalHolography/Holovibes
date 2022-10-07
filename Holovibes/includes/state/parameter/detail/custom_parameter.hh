#pragma once

#include "parameter.hh"

namespace holovibes
{
template <size_t N>
struct StringLiteral
{
    constexpr StringLiteral(const char (&str)[N]) { std::copy_n(str, N, value); }
    constexpr operator const char*() const { return value; }
    char value[N];
};

template <typename T, auto DefaultValue, StringLiteral Key, typename TConstRef = const T&>
class CustomParameter : public IParameter
{
  public:
    using ValueType = T;
    using TransfertType = TConstRef;

  public:
    CustomParameter()
        : value_(DefaultValue)
    {
    }

    CustomParameter(TransfertType value)
        : value_(value)
    {
    }
    virtual ~CustomParameter() override {}

  public:
    TransfertType get_value() const { return value_; }
    ValueType& get_value() { return value_; }
    void set_value(TransfertType value) { value_ = value; }

    static const char* static_key() { return Key; }
    const char* get_key() const override { return static_key(); }

  public:
    virtual void sync_with(IParameter* ref) override
    {
        const ValueType& new_value = reinterpret_cast<CustomParameter*>(ref)->get_value();
        if (value_ != new_value)
        {
            value_ = new_value;
            set_has_been_synchronized(true);
        }
    };

  protected:
    ValueType value_;
};

template <bool DefaultValue, StringLiteral Key>
using BoolParameter = CustomParameter<bool, DefaultValue, Key, bool>;

template <uint DefaultValue, StringLiteral Key>
using UIntParameter = CustomParameter<uint, DefaultValue, Key, uint>;

template <int DefaultValue, StringLiteral Key>
using IntParameter = CustomParameter<int, DefaultValue, Key, int>;

template <int DefaultValue, StringLiteral Key>
using FloatParameter = CustomParameter<float, DefaultValue, Key, float>;

template <StringLiteral DefaultValue, StringLiteral Key>
using StringParameter = CustomParameter<std::string, DefaultValue, Key>;

// using pomme = FloatParameter<1, "pomme">;

} // namespace holovibes
