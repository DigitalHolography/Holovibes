#pragma once

#include <unordered_map>
#include <type_traits>
#include "parameter.hh"

namespace holovibes
{

template <typename... T>
class StaticContainer;

template <>
class StaticContainer<>
{
  public:
    StaticContainer() {}

  public:
    template <typename StaticContainerRef>
    void sync_with(StaticContainerRef& ref)
    {
    }

    template <typename K>
    static constexpr bool has()
    {
        return false;
    }

    template <typename FunctionClass, typename... Args>
    void operator()(FunctionClass functions_class, Args&&... args)
    {
    }

    template <typename U>
    const U& get() const
    {
        static_assert(false, "Type is not in the template");
    }

    template <typename U>
    U& get()
    {
        static_assert(false, "Type is not in the template");
    }
};

template <typename TParameter, typename... R>
class StaticContainer<TParameter, R...> : public StaticContainer<R...>
{
  private:
    TParameter value_;

  public:
    StaticContainer()
        : StaticContainer<R...>()
        , value_()
    {
    }

  public:
    template <typename K>
    static constexpr bool has()
    {
        if (std::is_same_v<TParameter, K> == true)
            return true;
        return StaticContainer<R...>::template has<K>();
    }

  public:
    template <typename StaticContainerRef>
    void sync_with(StaticContainerRef& ref)
    {
        value_.sync_with(&ref.template get<TParameter>());
        StaticContainer<R...>::sync_with(ref);
    }

    template <typename FunctionClass, typename... Args>
    void operator()(FunctionClass& functions_class, Args&&... args)
    {
        functions_class.template operator()<TParameter>(value_, std::forward<Args>(args)...);
        StaticContainer<R...>::template operator()<FunctionClass>(functions_class, std::forward<Args>(args)...);
    }

    template <typename FunctionClass, typename... Args>
    void call(Args&&... args)
    {
        FunctionClass functions_class;
        this->operator()(functions_class, std::forward<Args>(args)...);
    }

  public:
    // getters
    template <typename U>
    requires std::is_same_v<TParameter, U>
    const U& get() const { return value_; }

    template <typename U>
    requires(false == std::is_same_v<TParameter, U>) const U& get() const
    {
        return StaticContainer<R...>::template get<U>();
    }

    template <typename U>
    requires std::is_same_v<TParameter, U> U& get() { return value_; }

    template <typename U>
    requires(false == std::is_same_v<TParameter, U>) U& get() { return StaticContainer<R...>::template get<U>(); }
};

} // namespace holovibes
