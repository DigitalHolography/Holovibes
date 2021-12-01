#pragma once

#if defined(_MSC_VER) && !defined(__clang__)
#include "map_macro_msvc.hh"
#else
#include "map_macro.hh"
#endif
#include "micro_cache.hh"
#include "checker.hh"

/*
 * This macro is the core of the micro-cache's functioning.
 * As explained in micro_cache.hh, the microcache needs to regularly synchronize with the GSH. In order to know which
 * variable(s) have changed during synchronization, we store the variables in a struct whose name is the variable's
 * name, and we generate a setter at compile time.
 * This setter is used in the commands received by the GSH ; while updating its own parameters, it will also store the
 * new values in every to_update field of the corresponding micro-caches variables.
 *
 * For example, consider this simple struct ExampleCache:
 *
 * NEW_MICRO_CACHE(ExampleCache, (int, a))
 *
 * At compile time, it will expand to:
 *
 *  struct ExampleCache : public MicroCache
 *  {
 *    friend class GSH;
 *    private:
 *      struct a_t
 *      {
 *          int obj;
 *          volatile bool to_update = nullptr;
 *      };
 *      a_t a;
 *
 *      void set_a(const int& _val)
 *      {
 *          a.obj = _val;
 *          trigger_a();
 *      }
 *
 *      int &get_a_ref() noexcept { return a.obj; }
 *
 *      void trigger_a()
 *      {
 *          for (MicroCache * cache : micro_caches_)
 *          {
 *              for (ExampleCache cache : micro_caches<ExampleCache>)
 *                  cache->a.to_update = true;
 *          }
 *      }
 *
 *    public:
 *      const int& get_a() const noexpect { return a.obj; }
 *
 *    ExampleCache(bool truth = false)
 *      : MicroCache(truth)
 *    {
 *        if (truth)
 *        {
 *            cache_truth<ExampleCache> = this;
 *            return;
 *        }
 *
 *        assert(cache_truth<ExampleCache> != nullptr);
 *        a.obj = cache_truth<ExampleCache>.a.obj;
 *        a.to_update = false;
 *        micro_caches<ExampleCache>.insert(this);
 *    }
 *
 *    ~ExampleCache()
 *    {
 *        if (truth_)
 *            cache_truth<ExampleCache> = nullptr;
 *        else
 *            micro_caches.erase(this);
 *    }
 *
 *    void synchronize() override
 *    {
          assert(cache_truth<ExampleCache> != this);
 *        if (a.to_update)
 *        {
 *            a.obj = cache_truth<ExampleCache>.a.obj;
 *            a.to_update = false;
 *        }
 *    }
 *  };
 *
 *  Note: for complex type parameters with commas in template parameters please use a 'using' directive
 */

#define __MONITORED_MEMBER(type, var)                                                                                  \
    struct var##_t                                                                                                     \
    {                                                                                                                  \
        type obj;                                                                                                      \
        volatile bool to_update;                                                                                       \
    };                                                                                                                 \
    var##_t var;                                                                                                       \
                                                                                                                       \
    void set_##var(const type& _val)                                                                                   \
    {                                                                                                                  \
        var.obj = _val;                                                                                                \
        trigger_##var();                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    type& get_##var##_ref() noexcept { return var.obj; }                                                               \
                                                                                                                       \
    void trigger_##var()                                                                                               \
    {                                                                                                                  \
        for (decltype(this) cache : micro_caches<decltype(*this)>)                                                     \
            cache->var.to_update = true;                                                                               \
    }                                                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    const type& get_##var() const noexcept { return var.obj; }

#ifndef MICRO_CACHE_DEBUG

#define _MONITORED_MEMBER(type, var)                                                                                   \
  private:                                                                                                             \
    __MONITORED_MEMBER(type, var)

#else

#define _MONITORED_MEMBER(type, var)                                                                                   \
  public:                                                                                                              \
    __MONITORED_MEMBER(type, var)

#endif

#define _SYNC_VAR(type, var)                                                                                           \
    var.obj = cache_truth<decltype(*this)>->var.obj;                                                                   \
    var.to_update = false;

#define _IF_NEED_SYNC_VAR(type, var)                                                                                   \
    if (var.to_update)                                                                                                 \
    {                                                                                                                  \
        var.obj = cache_truth<decltype(*this)>->var.obj;                                                               \
        var.to_update = false;                                                                                         \
    }

#define NEW_MICRO_CACHE(name, ...)                                                                                     \
    struct name : MicroCache                                                                                           \
    {                                                                                                                  \
        name(bool truth = false)                                                                                       \
            : MicroCache(truth)                                                                                        \
        {                                                                                                              \
            if (truth_)                                                                                                \
            {                                                                                                          \
                cache_truth<decltype(*this)> = this;                                                                   \
                return;                                                                                                \
            }                                                                                                          \
                                                                                                                       \
            CHECK(cache_truth<decltype(*this)> != nullptr) << "You must register a truth cache for class: " << #name;  \
                                                                                                                       \
            MAP(_SYNC_VAR, __VA_ARGS__)                                                                                \
            micro_caches<decltype(*this)>.insert(&(*this));                                                            \
        }                                                                                                              \
                                                                                                                       \
        ~name()                                                                                                        \
        {                                                                                                              \
            if (truth_)                                                                                                \
                cache_truth<decltype(*this)> = nullptr;                                                                \
            else                                                                                                       \
                micro_caches<decltype(*this)>.erase(&(*this));                                                         \
        }                                                                                                              \
                                                                                                                       \
        void synchronize() override                                                                                    \
        {                                                                                                              \
            CHECK(truth_ == false) << "You can't synchronize a truth cache";                                           \
            MAP(_IF_NEED_SYNC_VAR, __VA_ARGS__);                                                                       \
        }                                                                                                              \
                                                                                                                       \
        MAP(_MONITORED_MEMBER, __VA_ARGS__);                                                                           \
                                                                                                                       \
        friend class GSH;                                                                                              \
    };
