#pragma once

#include "micro_cache.hh"

#define MONITORED_MEMBER(type, var)                                                                                    \
  private:                                                                                                             \
    struct var##_t                                                                                                     \
    {                                                                                                                  \
        type obj;                                                                                                      \
        type* volatile to_update = nullptr;                                                                            \
    };                                                                                                                 \
    var##_t var;                                                                                                       \
                                                                                                                       \
    void set_##var(type _val)                                                                                          \
    {                                                                                                                  \
        var.obj = _val;                                                                                                \
        for (MicroCache * cache : micro_caches_)                                                                       \
        {                                                                                                              \
            decltype(this) underlying_cache = dynamic_cast<decltype(this)>(cache);                                     \
            if (underlying_cache == nullptr)                                                                           \
                return;                                                                                                \
                                                                                                                       \
            underlying_cache->var.to_update = &var.obj;                                                                \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    type get_##var() { return var.obj; }

namespace holovibes
{
template <class First>
void MicroCache::synchronize(First& elem)
{
    if (elem.to_update != nullptr)
    {
        elem.obj = *elem.to_update;
        elem.to_update = nullptr;
    }
}

template <class First, class... Args>
void MicroCache::synchronize(First& elem, Args&&... args)
{
    synchronize<First>(elem);
    synchronize<Args...>(std::forward<Args>(args)...);
}

} // namespace holovibes
