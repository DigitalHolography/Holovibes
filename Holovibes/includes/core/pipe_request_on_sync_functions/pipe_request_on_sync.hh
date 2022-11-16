#pragma once

#include "micro_cache.hh"
#include "logger.hh"

class Pipe;

namespace holovibes
{

#ifndef DISABLE_LOG_UPDATE_PIPE
#define LOG_UPDATE_PIPE(type) LOG_TRACE(compute_worker, "UPDATE " #type);
#else
#define LOG_UPDATE_PIPE(type)
#endif

class PipeRequestOnSync
{
  private:
    inline static bool requests_fail = false;
    inline static bool need_notify = true;
    inline static bool need_pipe_refresh = false;

  protected:
    static void request_fail()
    {
        LOG_ERROR(main, "Got a request fail in a pipe request");
        requests_fail = true;
    }
    static void request_pipe_refresh()
    {
        LOG_DEBUG(main, "Need a pipe refresh");
        need_pipe_refresh = true;
    }
    static void request_notify()
    {
        LOG_DEBUG(main, "Need notify");
        need_notify = true;
    }

  public:
    static void begin_requests()
    {
        requests_fail = false;
        need_pipe_refresh = false;
        need_notify = false;
    }

    static bool has_requests_fail() { return requests_fail; }
    static bool do_need_pipe_refresh() { return need_pipe_refresh; }
    static bool do_need_notify() { return need_notify; }
};
} // namespace holovibes
