#pragma once

#include "micro_cache.hh"

/*!
 * \brief This macro is the core of the micro-cache's functionning.
 * As explained in micro_cache.hh, the microcache needs to regularly synchronize with the GSH. In order to know which
 * variable(s) have changed during synchronization, we store the variables in a struct whose name is the variable's
 * name, and we generate a setter at compile time.
 * This setter is used in the commands received by the GSH ; while updating it's own parameters, it will also store the
 * new values in every to_update field of the corresponding micro-caches variables.
 *
 * For example, consider this simple struct ExampleCache:
 *
 *  struct ExampleCache : public MicroCache
 *  {
 *      MONITORED_MEMBER(int, a)
 *  };
 *
 * At compile time, it will expand to:
 *
 *  struct ExampleCache : public MicroCache
 *  {
 *    private:
 *      struct a_t
 *      {
 *          int obj;
 *          int* volatile to_update = nullptr;
 *      };
 *      a_t a;
 *
 *      void set_a(int _val)
 *      {
 *          a.obj = _val;
 *          for (MicroCache * cache : micro_caches_)
 *          {
 *              decltype(this) underlying_cache = dynamic_cast<decltype(this)>(cache);
 *              if (underlying_cache == nullptr)
 *                  return;
 *
 *              underlying_cache->a.to_update = &a.obj;
 *          }
 *      }
 *
 *    public: int get_a() { return a.obj; }
 *  };
 *
 *
 */

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
