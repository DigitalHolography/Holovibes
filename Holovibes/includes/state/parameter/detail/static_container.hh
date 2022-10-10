#pragma once

#include <unordered_map>
#include <type_traits>
#include "parameter.hh"

namespace holovibes
{

using MapKeyParams = std::unordered_map<std::string_view, IParameter*>;

template <typename... T>
class StaticContainer;

template <>
class StaticContainer<>
{
  public:
    StaticContainer(MapKeyParams&) {}

  public:
    void force_sync_with();

    template <typename FunctionClass, typename... Args>
    void call(FunctionClass functions_class, Args&&... args)
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

template <typename T, typename... R>
class StaticContainer<T, R...> : public StaticContainer<R...>
{
  private:
    T value_;

  public:
    StaticContainer(MapKeyParams& map_key_params)
        : StaticContainer<R...>(map_key_params)
        , value_()
    {
        map_key_params[T::static_key()] = &value_;
    }

  public:
    template <typename StaticContainerRef>
    void force_sync_with(StaticContainerRef& ref)
    {
        value_.sync_with(&ref.template get<T>());
        if constexpr (sizeof...(R) > 0)
            StaticContainer<R...>::force_sync_with(ref);
    }

    template <typename FunctionClass, typename... Args>
    void call(FunctionClass& functions_class, Args&&... args)
    {
        StaticContainer<R...>::template call<FunctionClass>(functions_class, std::forward<Args>(args)...);

        constexpr bool has_member_test = requires(FunctionClass functions_class)
        {
            functions_class.template test<T>(value_);
        };
        if constexpr (has_member_test) if (!functions_class.template test<T>(value_)) return;

        functions_class.template call<T>(value_, std::forward<Args>(args)...);
    }

  public:
    // getters
    template <typename U>
    requires std::is_same_v<T, U>
    const U& get() const { return value_; }

    template <typename U>
    requires std::is_same_v<T, U> U& get() { return value_; }

    template <typename U>
    requires(false == std::is_same_v<T, U>) const U& get() const { return StaticContainer<R...>::template get<U>(); }

    template <typename U>
    requires(false == std::is_same_v<T, U>) U& get() { return StaticContainer<R...>::template get<U>(); }
};

} // namespace holovibes
