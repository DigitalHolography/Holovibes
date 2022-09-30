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
};

template <typename T, typename... R>
class StaticContainer<T, R...> : public StaticContainer<R...>
{
  private:
    T value_;

  public:
    StaticContainer(MapKeyParams& map_key_params)
        : StaticContainer<R...>(map_key_params)
    {
        map_key_params[T::static_key()] = &value_;
    }

  public:
    void force_sync_with(StaticContainer<T, R...>& ref)
    {
        value_ = ref.value_;
        if constexpr (sizeof...(R) > 0)
            StaticContainer<R...>::force_sync_with(static_cast<StaticContainer<R...>>(ref));
    }

  public:
    template <typename U>
    requires std::is_same_v<T, U>
    const U& get() const { return value_; }

    template <typename U>
    requires std::is_same_v<T, U> U& get() { return value_; }

    template <typename U>
    requires(false == std::is_same_v<T, U>) const U& get() const { return StaticContainer<R...>::get<U>(); }

    template <typename U>
    requires(false == std::is_same_v<T, U>) U& get() { return StaticContainer<R...>::get<U>(); }

    template <typename U>
    requires std::is_same_v<T, U>
    void set(U&& value) { value_ = std::forward<U>(value); }

    template <typename U>
    requires(false == std::is_same_v<T, U>) void set(U&& value) { StaticContainer<R...>::set(std::forward<U>(value)); }
};

} // namespace holovibes
