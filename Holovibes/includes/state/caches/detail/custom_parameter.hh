#pragma once

#include <concepts>
#include <utility>

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

struct FloatLiteral
{
    constexpr FloatLiteral(float a) { value = a; }
    constexpr FloatLiteral(int a) { value = a; }
    constexpr operator float() const { return value; }
    float value;
};

template <typename T>
struct VectorLiteral
{
    constexpr operator std::vector<T>() const { return std::vector<T>{}; }
    static constexpr VectorLiteral instance() { return VectorLiteral(); }
};

template <typename T, auto DefaultValue, StringLiteral Key, typename TRef = const T&>
class CustomParameter : public IParameter
{
  public:
    using ValueType = T;
    using ConstRefType = TRef;

  public:
    CustomParameter()
        : value_(DefaultValue)
    {
    }

    CustomParameter(ConstRefType value)
        : value_(value)
    {
    }

    CustomParameter(const std::convertible_to<ValueType> auto& value)
        : value_(static_cast<ValueType>(value))
    {
    }

    virtual ~CustomParameter() override {}

    operator ConstRefType() const { return value_; }

  public:
    ConstRefType get_value() const { return value_; }
    ValueType& get_value() { return value_; }
    void set_value(ConstRefType value) { value_ = value; }

    static const char* static_key() { return Key; }
    const char* get_key() const override { return static_key(); }

  public:
    virtual void sync_with(IParameter* ref) override
    {
        CustomParameter* ref_cast = dynamic_cast<CustomParameter*>(ref);
        if (ref_cast == nullptr)
        {
            LOG_ERROR(main, "Not supposed to end here : Not the good type casted when syncing");
            return;
        }

        constexpr bool has_op_plus = requires(ValueType lhs, ValueType rhs) { lhs.operator+(rhs); };

        ValueType& new_value = ref_cast->get_value();
        if constexpr (has_op_plus)
        {
            if (value_ != new_value)
            {
                value_ = new_value;
                set_has_been_synchronized(true);
            }
        }
        else
        {
            LOG_WARN(main, "Couldn't check if the value has been changed");
            value_ = new_value;
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

template <FloatLiteral DefaultValue, StringLiteral Key>
using FloatParameter = CustomParameter<float, DefaultValue, Key, float>;

template <StringLiteral DefaultValue, StringLiteral Key>
using StringParameter = CustomParameter<std::string, DefaultValue, Key>;

template <typename T, StringLiteral Key>
using VectorParameter = CustomParameter<std::vector<T>, VectorLiteral<T>::instance(), Key>;

// using pomme = FloatParameter<1, "pomme">;

} // namespace holovibes
