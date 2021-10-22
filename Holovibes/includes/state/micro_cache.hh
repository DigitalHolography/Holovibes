#pragma once

#include <map>
#include <set>
#include <type_traits>
namespace holovibes
{

struct MicroCache
{
    MicroCache();

    ~MicroCache();

    virtual void synchronize() = 0;

  protected:
    template <class First, class... Args>
    void synchronize(First& elem, Args&&... args);

    template <class First>
    void synchronize(First& elem);

    static inline std::set<MicroCache*> micro_caches_;
};
} // namespace holovibes

#include "micro_cache.hxx"
