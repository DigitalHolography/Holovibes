#pragma once

#include "micro_cache.hh"
#include "logger.hh"
#include "all_caches.hh"
#include "pipe_sync_front_end.hh"

namespace holovibes
{

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
};
} // namespace holovibes
