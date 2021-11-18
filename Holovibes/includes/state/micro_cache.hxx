#pragma once

#include "micro_cache.hh"
#include "map_macro.hh"
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
 *  struct ExampleCache
 *  {
 *      using ref_t = Ref;
 *      using cache_t = Cache;
 *
 *      struct Ref : MicroCache
 *      {
 *          Ref() { cache_truth<ref_t> = this; }
 *          ~Ref() { cache_truth<ref_t> = nullptr; }
 *
 *          void set_a(int _val)
 *          {
 *               a = _val;
 *               trigger_a();
 *          }
 *
 *          int &get_a() noexcept { return a; }
 *          int get_a() noexcept const { return a; }
 *
 *          void trigger_a()
 *          {
 *              for (MicroCache * cache : micro_caches_)
 *              {
 *                  for (ExampleCache cache : micro_caches<cache_t>)
 *                      cache->a.to_update = true;
 *              }
 *          }
 *        private:
 *          int a;
 *      };
 *
 *      struct Cache
 *      {
 *          Cache()
 *          {
 *              a.obj = cache_truth<ref_t>.a.obj;
 *              a.to_update = false;
 *              micro_caches<cache_t>.insert(this);
 *          }
 *
 *          ~Cache() { micro_caches<cache_t>.erase(this); }
 *
 *          int get_a() noexcept const { return a.obj; }
 *
 *          void synchronize() noexcept
 *          {
 *              if (a.to_update)
 *              {
 *                  a.obj = cache_truth<ref_t>.a.obj;
 *                  a.to_update = false;
 *              }
 *          }
 *
 *        private:
 *          struct a_t
 *          {
 *              int obj;
 *              volatile bool to_update = nullptr;
 *          };
 *          a_t a;
 *      };
 *  };
 *
 *  Note: for complex type parameters with commas in template parameters please use a 'using' directive
 */

#ifdef _DEBUG
#define LOG_UPDATE(var) LOG_DEBUG << "Update " << #var << " : " << var.obj << " -> " << cache_truth<ref_t>->var;
#else
#define LOG_UPDATE(var)
#endif

#define _SYNC_VAR(type, var)                                                                                           \
    var.obj = cache_truth<ref_t>->var;                                                                                 \
    var.to_update = false;

#define _IF_NEED_SYNC_VAR(type, var)                                                                                   \
    if (var.to_update)                                                                                                 \
    {                                                                                                                  \
        LOG_UPDATE(var)                                                                                                \
        var.obj = cache_truth<ref_t>->var;                                                                             \
        var.to_update = false;                                                                                         \
    }

#define _DEFINE_VAR(type, var) type var;

#define _DEFINE_CACHE_VAR(type, var)                                                                                   \
    struct var##_t                                                                                                     \
    {                                                                                                                  \
        type obj;                                                                                                      \
        volatile bool to_update;                                                                                       \
    };                                                                                                                 \
    var##_t var;

#define _GETTER(type, var)                                                                                             \
    inline type get_##var() const noexcept { return var.obj; }

#define _GETTER_SETTER_TRIGGER(type, var)                                                                              \
    void set_##var(const type& _val)                                                                                   \
    {                                                                                                                  \
        var = _val;                                                                                                    \
        trigger_##var();                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    type get_##var() const noexcept { return var; }                                                                    \
                                                                                                                       \
    type& get_##var##_ref() noexcept { return var; }                                                                   \
                                                                                                                       \
    void trigger_##var()                                                                                               \
    {                                                                                                                  \
        for (cache_t * cache : micro_caches<cache_t>)                                                                  \
            cache->var.to_update = true;                                                                               \
    }

#define NEW_MICRO_CACHE(name, ...)                                                                                     \
    struct name                                                                                                        \
    {                                                                                                                  \
        struct Ref;                                                                                                    \
        struct Cache;                                                                                                  \
        using ref_t = Ref;                                                                                             \
        using cache_t = Cache;                                                                                         \
        struct Ref : MicroCache                                                                                        \
        {                                                                                                              \
          private:                                                                                                     \
            MAP(_DEFINE_VAR, __VA_ARGS__);                                                                             \
                                                                                                                       \
          public:                                                                                                      \
            Ref() { cache_truth<ref_t> = this; }                                                                       \
            ~Ref() { cache_truth<ref_t> = nullptr; }                                                                   \
                                                                                                                       \
            MAP(_GETTER_SETTER_TRIGGER, __VA_ARGS__);                                                                  \
            friend struct Cache;                                                                                       \
        };                                                                                                             \
                                                                                                                       \
        struct Cache : MicroCache                                                                                      \
        {                                                                                                              \
          private:                                                                                                     \
            MAP(_DEFINE_CACHE_VAR, __VA_ARGS__);                                                                       \
                                                                                                                       \
          public:                                                                                                      \
            Cache()                                                                                                    \
            {                                                                                                          \
                CHECK(cache_truth<ref_t> != nullptr) << "You must register a truth cache for class: " << #name;        \
                MAP(_SYNC_VAR, __VA_ARGS__);                                                                           \
                micro_caches<cache_t>.insert(this);                                                                    \
            }                                                                                                          \
            ~Cache() { micro_caches<cache_t>.erase(this); }                                                            \
                                                                                                                       \
            void synchronize() { MAP(_IF_NEED_SYNC_VAR, __VA_ARGS__); }                                                \
                                                                                                                       \
            MAP(_GETTER, __VA_ARGS__);                                                                                 \
            friend struct Ref;                                                                                         \
        };                                                                                                             \
    };
