#pragma once

namespace holovibes
{

struct MicroCache
{
    MicroCache(bool truth);

    virtual void synchronize() = 0;

  protected:
    const bool truth_;
    template <class T>
    static inline std::remove_reference_t<T>* cache_truth;

    template <class T>
    static inline std::set<std::remove_reference_t<T>*> micro_caches;
};
} // namespace holovibes

#include "micro_cache.hxx"
