#pragma once

#include "micro_cache.hh"
#include "logger.hh"
#include <functional>
#include <array>
#include "all_caches.hh"

class Pipe;

namespace holovibes
{

class FrontEndMethodsCallback
{
  public:
    static void set(std::function<void(std::function<void(void)>&)> f) { callback_ = f; }
    static void call(std::function<void(void)> f)
    {
        if (callback_)
            return callback_(f);
        else
            return f();
    }

  private:
    static inline std::function<void(std::function<void(void)>&)> callback_;
};

template <typename PipeRequest, typename FrontEndMethods>
class PipeRequestOnSyncWrapper
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType new_value, Pipe& pipe)
    {
        FrontEndMethods method;
        FrontEndMethodsCallback::call([&]() { method.template before_method<T>(); });
        PipeRequest pipe_request;
        pipe_request.template operator()<T>(new_value, pipe);
        FrontEndMethodsCallback::call([&]() { method.template after_method<T>(); });
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, typename T::ConstRefType old_value, Pipe& pipe)
    {
        FrontEndMethods method;
        FrontEndMethodsCallback::call([&]() { method.template before_method<T>(); });
        PipeRequest pipe_request;
        pipe_request.template on_sync<T>(new_value, old_value, pipe);
        FrontEndMethodsCallback::call([&]() { method.template after_method<T>(); });
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

class PipeRequestOnSync
{
  private:
    inline static bool requests_fail = false;
    inline static bool need_notify = true;
    inline static bool need_pipe_refresh = false;
    inline static bool clean_pipe = false;

  protected:
    static void request_fail()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_ERROR("Got a request fail in a pipe request");
#endif
        requests_fail = true;
    }
    static void request_pipe_refresh()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_DEBUG("Need a pipe refresh");
#endif
        need_pipe_refresh = true;
    }

    static void disable_pipe()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_DEBUG("disable pipe refresh");
#endif
        clean_pipe = true;
    }
    static void request_notify()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_DEBUG("Need notify");
#endif
        need_notify = true;
    }

  public:
    static void begin_requests()
    {
        clean_pipe = false;
        requests_fail = false;
        need_pipe_refresh = false;
        need_notify = true;
    }

    static bool has_requests_fail() { return requests_fail; }
    static bool do_need_pipe_refresh() { return need_pipe_refresh; }
    static bool do_need_notify() { return need_notify; }
    static bool do_disable_pipe() { return clean_pipe; }
};
} // namespace holovibes
