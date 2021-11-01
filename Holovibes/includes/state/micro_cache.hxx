#pragma once

#include "micro_cache.hh"
#include "map_macro.hh"
#include "checker.hh"

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
 *              decltype(this) underlying_cache = dynamic_cast<decltype(this)>(cache);
 *              if (this != cache || underlying_cache == nullptr)
 *                  continue;
 *
 *              underlying_cache->a.to_update = &a.obj;
 *          }
 *      }
 *
 *    public:
 *      const int& get_a() const noexpect { return a.obj; }
 *  };
 *
 *  Note: for complex type parameters with commas in template parameters please use a 'using' directive
 *
 */

#ifndef MICRO_CACHE_DEBUG

#define MONITORED_MEMBER(type, var)                                                                                    \
  private:                                                                                                             \
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

#else

#define MONITORED_MEMBER(type, var)                                                                                    \
  public:                                                                                                              \
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
    const type& get_##var() const noexcept { return var.obj; }

#endif

#define SYNC_VAR(type, var)                                                                                            \
    var.obj = cache_truth<decltype(*this)>->var.obj;                                                                   \
    var.to_update = false;

#define IF_NEED_SYNC_VAR(type, var)                                                                                    \
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
            MAP(SYNC_VAR, __VA_ARGS__)                                                                                 \
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
        void synchronize() override { MAP(IF_NEED_SYNC_VAR, __VA_ARGS__); }                                            \
                                                                                                                       \
        MAP(MONITORED_MEMBER, __VA_ARGS__);                                                                            \
                                                                                                                       \
        friend class GSH;                                                                                              \
    };
