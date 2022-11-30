#pragma once

#include <concepts>
#include <utility>

#include "Iparameter.hh"

namespace holovibes
{

namespace detail
{
template <typename T>
inline bool has_value_change(typename T::ConstValueRef old_value, typename T::ConstValueRef new_value)
{
    constexpr bool has_op_neq = requires(typename T::ConstValueRef lhs, typename T::ConstValueRef rhs) { lhs != rhs; };

    if constexpr (has_op_neq) return value_ != new_value;

    LOG_WARN("Couldn't check if the value has been change ; T = {}", typeid(T).name());
    return true;
}

} // namespace detail

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
struct DefaultLiteral
{
    constexpr DefaultLiteral() {}
    constexpr operator T() const { return T{}; }
};

template <typename T, auto DefaultValue, StringLiteral Key, typename TRef = const T&>
class Parameter : public IParameter
{
  public:
    using ValueType = T;
    using ConstRefType = TRef;

  public:
    Parameter()
        : value_(DefaultValue)
    {
    }

    Parameter(ConstRefType value)
        : value_(value)
    {
    }

    Parameter(const std::convertible_to<ValueType> auto& value)
        : value_(static_cast<ValueType>(value))
    {
    }

    virtual ~Parameter() override {}

    operator ConstRefType() const { return value_; }

  public:
    ConstRefType get_value() const { return value_; }
    ValueType& get_value() { return value_; }
    void set_value(ConstRefType value) { value_ = value; }

    const char* get_key() const override { return Key; }

  public:
    inline bool has_parameter_change_valuetype(ConstRefType ref) const
    {
        constexpr bool has_op_neq = requires(ValueType lhs, ConstRefType rhs) { lhs != rhs; };

        if constexpr (has_op_neq) return value_ != ref;

        LOG_WARN("Couldn't check if the value has been change ; T = {}", typeid(T).name());
        return true;
    }

    virtual bool has_parameter_change(IParameter* ref) const override
    {
        Parameter* ref_cast = dynamic_cast<Parameter*>(ref);
        if (ref_cast == nullptr)
        {
            LOG_ERROR("Not supposed to end here : Not the good type casted when syncing");
            return true;
        }

        ConstRefType new_value = ref_cast->get_value();
        return has_parameter_change_valuetype(new_value);
    };

    virtual void sync_with(IParameter* ref) override
    {
        if (has_parameter_change(ref))
        {
            // technically doesn't need to check here
            Parameter* ref_cast = reinterpret_cast<Parameter*>(ref);
            value_ = ref_cast->get_value();
            set_has_been_synchronized(true);
        }
    };

  protected:
    ValueType value_;
};

// T is a class that herit from IParameter like Parameter, ...
template <typename T>
class DuplicatedParameter : public IDuplicatedParameter
{
  public:
    using BaseParameter = T;

  public:
    typename BaseParameter::ValueType& get_value() { return value_; }
    void set_value(typename BaseParameter::ValueType& value) { value_ = value; }
    virtual void save_current_value(const IParameter* param) override
    {
        const T* param_as_t = dynamic_cast<const T*>(param);
        if (param_as_t == nullptr)
        {
            LOG_ERROR("Not supposed to end here : Not the good type casted when syncing");
            return;
        }

        value_ = param_as_t->get_value();
    }

  protected:
    typename BaseParameter::ValueType value_;
};

template <bool DefaultValue, StringLiteral Key>
using BoolParameter = Parameter<bool, DefaultValue, Key, bool>;

template <uint DefaultValue, StringLiteral Key>
using UIntParameter = Parameter<uint, DefaultValue, Key, uint>;

template <int DefaultValue, StringLiteral Key>
using IntParameter = Parameter<int, DefaultValue, Key, int>;

template <FloatLiteral DefaultValue, StringLiteral Key>
using FloatParameter = Parameter<float, DefaultValue, Key, float>;

template <StringLiteral DefaultValue, StringLiteral Key>
using StringParameter = Parameter<std::string, DefaultValue, Key>;

template <typename T, StringLiteral Key>
using VectorParameter = Parameter<std::vector<T>, DefaultLiteral<std::vector<T>>{}, Key>;

struct TriggerRequest
{
};

template <StringLiteral Key>
using TriggerParameter = Parameter<TriggerRequest, DefaultLiteral<TriggerRequest>{}, Key, TriggerRequest>;

} // namespace holovibes
