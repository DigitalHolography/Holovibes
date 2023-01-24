#pragma once

#include "micro_cache.hh"

// FIXME - DELETE OR IMPLEM

template <typename... Params>
class MicroCache<Params...>::BasicMicroCache
{
  public:
    BasicMicroCache()
        : key_container_{}
        , container_{}
    {
        container_.template call<FillMapKeyParams>(key_container_);
    }

  public:
    template <typename FunctionClass, typename... Args>
    void call(Args&&... args)
    {
        container_.template call<FunctionClass>(std::forward<Args>(args)...);
    }

  public:
    const MapKeyParams& get_map_key() const { return key_container_; }
    MapKeyParams& get_map_key() { return key_container_; }

    StaticContainer<Params...>& get_container() { return container_; }

  public:
    template <typename T>
    typename T::ConstRefType get_value() const
    {
        return container_.template get<T>().get_value();
    }

    template <typename T>
    static constexpr bool has()
    {
        return StaticContainer<Params...>::template has<T>();
    }

  protected:
    template <typename T>
    T& get_type()
    {
        return container_.template get<T>();
    }

  protected:
    MapKeyParams key_container_;
    StaticContainer<Params...> container_;
};