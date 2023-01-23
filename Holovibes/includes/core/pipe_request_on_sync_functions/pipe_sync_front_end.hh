#pragma once

#include "micro_cache.hh"
#include "logger.hh"
#include <functional>
#include <array>
#include "all_caches.hh"

class Pipe;

// FIXME : maybe in namespace api ?
namespace holovibes
{

class FrontEndCallbackOnSync
{
  public:
    static void set_caller(std::function<void(std::function<void(void)>&)> f) { caller_ = f; }
    static void call(std::function<void(void)> f)
    {
        if (caller_)
            return caller_(f);
        else
            return f();
    }

    static void set_lock_front_end(std::function<void(void)> f) { lock_front_end_ = f; }
    static void call_lock_front_end()
    {
        if (lock_front_end_)
            return lock_front_end_();
    }

    static void set_unlock_front_end(std::function<void(void)> f) { unlock_front_end_ = f; }
    static void call_unlock_front_end()
    {
        if (unlock_front_end_)
            return unlock_front_end_();
    }

  private:
    static inline std::function<void(std::function<void(void)>&)> caller_;

    static inline std::function<void(void)> lock_front_end_;
    static inline std::function<void(void)> unlock_front_end_;
};

template <typename PipeRequest, typename FrontEndMethods>
class PipeSyncFrontEndWrapper
{
  public:
    void lock_front_end() { FrontEndCallbackOnSync::call_lock_front_end(); }
    void unlock_front_end() { FrontEndCallbackOnSync::call_unlock_front_end(); }

    template <typename T>
    void operator()(typename T::ConstRefType new_value, Pipe& pipe)
    {
        FrontEndMethods method;
        FrontEndCallbackOnSync::call([&]() { method.template before_method<T>(); });
        PipeRequest pipe_request;
        pipe_request.template operator()<T>(new_value, pipe);
        FrontEndCallbackOnSync::call([&]() { method.template after_method<T>(); });
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, typename T::ConstRefType old_value, Pipe& pipe)
    {
        FrontEndMethods method;
        FrontEndCallbackOnSync::call([&]() { method.template before_method<T>(); });
        PipeRequest pipe_request;
        pipe_request.template on_sync<T>(new_value, old_value, pipe);
        FrontEndCallbackOnSync::call([&]() { method.template after_method<T>(); });
    }
};

template <typename T>
class PipeRequestFrontEndMethods;

template <typename... MicroCacheParams>
class PipeRequestFrontEndMethods<MicroCache<MicroCacheParams...>>
{
  public:
    template <typename T>
    static void set_after_method(std::function<void(void)> after_method)
    {
        after_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()] = after_method;
    }

    template <typename T>
    static void set_before_method(std::function<void(void)> before_method)
    {
        before_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()] = before_method;
    }

  private:
    template <typename FunctionsClass, typename Parameter, typename... RestParameters>
    static void link_front_end_rec()
    {
        set_before_method<Parameter>(FunctionsClass::template before_method<Parameter>);
        set_after_method<Parameter>(FunctionsClass::template after_method<Parameter>);

        if constexpr (sizeof...(RestParameters) > 0)
            link_front_end_rec<FunctionsClass, RestParameters...>();
    }

  public:
    template <typename FunctionsClass>
    static void link_front_end()
    {
        link_front_end_rec<FunctionsClass, MicroCacheParams...>();
    }

  public:
    template <typename T>
    static void after_method()
    {
        if (after_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()])
        {
            after_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()]();
        }
    }

    template <typename T>
    static void before_method()
    {
        if (before_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()])
        {
            before_methods[MicroCache<MicroCacheParams...>::template get_index_of<T>()]();
        }
    }

  private:
    static inline std::array<std::function<void(void)>, MicroCache<MicroCacheParams...>::size()> before_methods;
    static inline std::array<std::function<void(void)>, MicroCache<MicroCacheParams...>::size()> after_methods;
};

// clang-format off
class AdvancedCacheFrontEndMethods : public PipeRequestFrontEndMethods<AdvancedCache>{};
class ComputeCacheFrontEndMethods : public PipeRequestFrontEndMethods<ComputeCache>{};
class ImportCacheFrontEndMethods : public PipeRequestFrontEndMethods<ImportCache>{};
class ExportCacheFrontEndMethods : public PipeRequestFrontEndMethods<ExportCache>{};
class CompositeCacheFrontEndMethods : public PipeRequestFrontEndMethods<CompositeCache>{};
class ViewCacheFrontEndMethods : public PipeRequestFrontEndMethods<ViewCache>{};
class ZoneCacheFrontEndMethods : public PipeRequestFrontEndMethods<ZoneCache>{};
// clang-format on

} // namespace holovibes
