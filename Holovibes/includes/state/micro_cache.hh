#pragma once

namespace holovibes
{

struct MicroCache;

template <class T>
concept MicroCacheDerived = std::is_base_of<MicroCache, std::remove_reference_t<T>>::value;

struct MicroCache
{
    MicroCache(bool truth);

    virtual void synchronize() = 0;

  protected:
    const bool truth_;
    template <MicroCacheDerived T>
    static inline std::remove_reference_t<T>* cache_truth;

    template <MicroCacheDerived T>
    static inline std::set<std::remove_reference_t<T>*> micro_caches;
};
} // namespace holovibes

#include "micro_cache.hxx"
