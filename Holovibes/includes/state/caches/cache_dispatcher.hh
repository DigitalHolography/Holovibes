#pragma once

namespace holovibes
{

template <typename... Caches>
class CacheDispatcher;

template <typename CacheHandle, typename... CacheOthers>
class CacheDispatcher<CacheHandle, CacheOthers...> : public CacheDispatcher<CacheOthers...>
{
  private:
    CacheHandle& cache_;

  public:
    CacheDispatcher<CacheHandle, CacheOthers...>(CacheHandle& handle, CacheOthers&... others)
        : CacheDispatcher<CacheOthers...>(std::forward<CacheOthers>(others)...)
        , cache_(handle)
    {
    }

  public:
    template <typename T>
    requires(true == CacheHandle::template has<T>()) auto& get() { return cache_; }

    template <typename T>
    requires(false == CacheHandle::template has<T>()) auto& get()
    {
        return CacheDispatcher<CacheOthers...>::template get<T>();
    }
};

template <>
class CacheDispatcher<>
{
  public:
    CacheDispatcher<>() {}

  public:
    template <typename T>
    void get()
    {
        static_assert(false, "Can't dispatch to this Attribute because it is not in this class");
    }
};

}; // namespace holovibes
