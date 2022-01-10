#pragma once

#include <memory>
#include <new>

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
 *  struct ExampleCache
 *  {
 * 		struct Ref;
 * 		struct Cache;
 *      using ref_t = Ref;
 *      using cache_t = Cache;
 *
 *      struct Ref : MicroCache
 *      {
 *        private:
 *          int a;
 * 
 * 		  public:
 *          Ref() { cache_reference<ref_t> = this; }
 *          ~Ref() { cache_reference<ref_t> = nullptr; }
 *
 *          void set_a(int _val)
 *          {
 *             	a = _val;
 *             	trigger_a();
 *         	}
 *
 *          int get_a() const noexcept { return a; }
 *         	int& get_a_ref() noexcept { return std::shared_ptr<int>{&a, [&](int*) { trigger_a(); }};
 *         	const int& get_a_const_ref() const noexcept { return a; }
 *
 *         	void trigger_a()
 *          {
 *          	for (cache_t * cache : micro_caches<cache_t>)
 *             	{
 *                 	for (cache_t cache : micro_caches<cache_t>)
 *                     	cache->a.to_update = true;
 *             	}
 *         	}
 * 
 * 			Ref(const Ref&) = delete;                                                                                  \
 *			Ref& operator=(const Ref&) = delete;                                                                       \
 *			Ref() { <ref_t> = this; }                                                                       \
 *			~Ref() { cache_reference<ref_t> = nullptr; }
 *
 * 			friend struct Cache;  
 *      };
 *
 *      struct Cache : MicroCache
 *      {
 * 		  private:
 *          struct a_t
 *          {
 *              int obj;
 *              volatile bool to_update;
 *          };
 *          a_t a;
 * 
 * 		  public:
 * 			Cache(const Cache&) = delete;                                                                              \
 *			Cache& operator=(const Cache&) = delete;
 *          Cache()
 *          {
 *              a.obj = <ref_t>.a.obj;
 *              a.to_update = false;
 *              micro_caches<cache_t>.insert(this);
 *          }
 *
 *          ~Cache() { micro_caches<cache_t>.erase(this); }
 *
 *          int get_a() noexcept const { return a.obj; }
 *
 *          void synchronize()
 *          {
 *              if (a.to_update)
 *              {
 *                  a.obj = cache_reference<ref_t>->a;
 *                  a.to_update = false;
 *              }
 *          }
 * 
 * 			friend struct Ref;
 *      };
 *  };
 *
 *  Note: for complex type parameters with commas in template parameters please use a 'using' directive
 */

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    return os << "vector";
}

#ifdef _DEBUG
#define LOG_UPDATE(var) LOG_DEBUG << "Update " << #var << " : " << var.obj << " -> " << cache_reference<ref_t>->var;
#else
#define LOG_UPDATE(var)
#endif

#define _SYNC_VAR(type, var)                                                                                           \
    var.obj = cache_reference<ref_t>->var;                                                                                 \
    var.to_update = false;

#define _IF_NEED_SYNC_VAR(type, var)                                                                                   \
    if (var.to_update)                                                                                                 \
    {                                                                                                                  \
        LOG_UPDATE(var)                                                                                                \
        var.obj = cache_reference<ref_t>->var;                                                                             \
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
    type& get_##var##_ref() noexcept                                                                                   \
    {                                                                                                                  \
        return std::shared_ptr<type>{&var, [&](type*) { trigger_##var(); }};                                           \
    }                                                                                                                  \
                                                                                                                       \
    const type& get_##var##_const_ref() const noexcept { return var; }                                                 \
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
            Ref() { cache_reference<ref_t> = this; }                                                                       \
            ~Ref() { cache_reference<ref_t> = nullptr; }                                                                   \
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
                CHECK(cache_reference<ref_t> != nullptr) << "You must register a reference cache for class: " << #name;        \
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

/*---------------------------------------------------------------------*/

#define _SYNC_VAR_INIT(type, var, val)                                                                                 \
    var.obj = cache_reference<ref_t>->var;                                                                                 \
    var.to_update = false;

#define _IF_NEED_SYNC_VAR_INIT(type, var, val)                                                                         \
    if (var.to_update)                                                                                                 \
    {                                                                                                                  \
        var.obj = cache_reference<ref_t>->var;                                                                             \
        var.to_update = false;                                                                                         \
    }

#define _DEFINE_VAR_INIT(type, var, val) alignas(std::hardware_constructive_interference_size) type var = val;

#define _DEFINE_CACHE_VAR_INIT(type, var, val)                                                                         \
    struct var##_t                                                                                                     \
    {                                                                                                                  \
        type obj = val;                                                                                                \
        volatile bool to_update;                                                                                       \
    };                                                                                                                 \
    var##_t var;

#define _GETTER_INIT(type, var, val)                                                                                   \
    inline type get_##var() const noexcept { return var.obj; }                                                         \
    inline const auto& get_##var##_const_ref() const noexcept { return var.obj; }

#define _GETTER_SETTER_TRIGGER_INIT(type, var, val)                                                                    \
    inline void set_##var(const type& _val)                                                                            \
    {                                                                                                                  \
        var = _val;                                                                                                    \
        trigger_##var();                                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    /* inline prevents MSVC from brain-dying, dunno why */                                                             \
    inline type get_##var() const noexcept { return var; }                                                             \
                                                                                                                       \
    /* inline prevents MSVC from brain-dying, dunno why */                                                             \
    inline const auto& get_##var##_const_ref() const noexcept { return var; }                                          \
                                                                                                                       \
    void trigger_##var()                                                                                               \
    {                                                                                                                  \
        for (cache_t * cache : micro_caches<cache_t>)                                                                  \
            cache->var.to_update = true;                                                                               \
    }                                                                                                                  \
                                                                                                                       \
    /* inline prevents MSVC from brain-dying, dunno why */																\
	/* this shared pointer is used so that the trigger is called at the destruction of the pointer, when the value is sure to be modified*/                                                             \
    inline auto get_##var##_ref()                                                                                      \
    {                                                                                                                  \
        return std::shared_ptr<type>{&var, [&](type*) { trigger_##var(); }};                                           \
    }

#define NEW_INITIALIZED_MICRO_CACHE(name, ...)                                                                         \
    struct name                                                                                                        \
    {                                                                                                                  \
        struct Ref;                                                                                                    \
        struct Cache;                                                                                                  \
        using ref_t = Ref;                                                                                             \
        using cache_t = Cache;                                                                                         \
        struct Ref : MicroCache                                                                                        \
        {                                                                                                              \
          private:                                                                                                     \
            MAP(_DEFINE_VAR_INIT, __VA_ARGS__);                                                                        \
                                                                                                                       \
          public:                                                                                                      \
            MAP(_GETTER_SETTER_TRIGGER_INIT, __VA_ARGS__);                                                             \
                                                                                                                       \
            Ref(const Ref&) = delete;                                                                                  \
            Ref& operator=(const Ref&) = delete;                                                                       \
            Ref() { cache_reference<ref_t> = this; }                                                                       \
            ~Ref() { cache_reference<ref_t> = nullptr; }                                                                   \
                                                                                                                       \
            friend struct Cache;                                                                                       \
        };                                                                                                             \
                                                                                                                       \
        struct Cache : MicroCache                                                                                      \
        {                                                                                                              \
          private:                                                                                                     \
            MAP(_DEFINE_CACHE_VAR_INIT, __VA_ARGS__);                                                                  \
                                                                                                                       \
          public:                                                                                                      \
            Cache(const Cache&) = delete;                                                                              \
            Cache& operator=(const Cache&) = delete;                                                                   \
            Cache()                                                                                                    \
            {                                                                                                          \
                CHECK(cache_reference<ref_t> != nullptr) << "You must register a reference cache for class: " << #name;        \
                MAP(_SYNC_VAR_INIT, __VA_ARGS__);                                                                      \
                micro_caches<cache_t>.insert(this);                                                                    \
            }                                                                                                          \
            ~Cache() { micro_caches<cache_t>.erase(this); }                                                            \
                                                                                                                       \
            void synchronize() { MAP(_IF_NEED_SYNC_VAR_INIT, __VA_ARGS__); }                                           \
                                                                                                                       \
            MAP(_GETTER_INIT, __VA_ARGS__);                                                                            \
            friend struct Ref;                                                                                         \
        };                                                                                                             \
    };
