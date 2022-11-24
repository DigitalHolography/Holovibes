#pragma once

#include "micro_cache.hh"
#include "logger.hh"
#include <functional>
#include <array>

class Pipe;

namespace holovibes
{

#ifndef DISABLE_LOG_UPDATE_PIPE
#define LOG_UPDATE_PIPE(type) LOG_TRACE(compute_worker, "UPDATE " #type);
#else
#define LOG_UPDATE_PIPE(type)
#endif

template <typename PipeRequest, typename AfterMethod>
class PipeRequestOnSyncWrapper
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType new_value, Pipe& pipe)
    {
        PipeRequest pipe_request;
        pipe_request.template operator()<T>(new_value, pipe);
        AfterMethod after_method;
        after_method.template operator()<T>();
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, typename T::ConstRefType old_value, Pipe& pipe)
    {
        PipeRequest pipe_request;
        pipe_request.template on_sync<T>(new_value, pipe);
        AfterMethod after_method;
        after_method.template operator()<T>();
    }
};

template <typename MicroCache>
class PipeRequestAfterMethod
{
  public:
    template <typename T>
    static void operartor()()
    {
        if (after_methods[MicroCache::get_index_of<T>()])
        {
          after_methods[MicroCache::get_index_of<T>()]();
        }
    }

    template <typename T>
    static void set_after_method()(std::function<void(void)> after_method)
    {
        after_methods[MicroCache::get_index_of<T>()] = after_method;
    }

  private:
    static std::array<std::function<void(void)>, MicroCache::size()> after_methods;

};
class AdvancedCacheAfterMethod : public PipeRequestAfterMethod<AdvancedCache>{};
class ComputeCacheAfterMethod : public PipeRequestAfterMethod<ComputeCache>{};
class ImportCacheAfterMethod : public PipeRequestAfterMethod<ImportCache>{};
class ExportCacheAfterMethod : public PipeRequestAfterMethod<ExportCache>{};
class CompositeCacheAfterMethod : public PipeRequestAfterMethod<CompositeCache>{};
class ViewCacheAfterMethod : public PipeRequestAfterMethod<ViewCache>{};
class ZoneCacheAfterMethod : public PipeRequestAfterMethod<ZoneCache>{};

class PipeRequestOnSync
{
  private:
    inline static bool requests_fail = false;
    inline static bool need_notify = true;
    inline static bool need_pipe_refresh = false;

  protected:
    static void request_fail()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_ERROR(main, "Got a request fail in a pipe request");
#endif
        requests_fail = true;
    }
    static void request_pipe_refresh()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_DEBUG(main, "Need a pipe refresh");
#endif
        need_pipe_refresh = true;
    }
    static void request_notify()
    {
#ifndef DISABLE_LOG_PIPE
        LOG_DEBUG(main, "Need notify");
#endif
        need_notify = true;
    }

  public:
    static void begin_requests()
    {
        requests_fail = false;
        need_pipe_refresh = false;
        need_notify = true;
    }

    static bool has_requests_fail() { return requests_fail; }
    static bool do_need_pipe_refresh() { return need_pipe_refresh; }
    static bool do_need_notify() { return need_notify; }
};
} // namespace holovibes
