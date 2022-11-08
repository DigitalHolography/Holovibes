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
    inline static bool pipe_need_to_refresh = false;

  protected:
    static void request_fail()
    {
        LOG_ERROR(main, "Got a request fail in a pipe request");
        requests_fail = true;
    }
    static void request_pipe_refresh()
    {
        LOG_DEBUG(main, "Need a pipe refresh");
        pipe_need_to_refresh = true;
    }

  public:
    static void begin_requests()
    {
        requests_fail = false;
        pipe_need_to_refresh = false;
    }
    static bool has_requests_fail() { return requests_fail; }
    static bool do_need_pipe_refresh() { return pipe_need_to_refresh; }
};
} // namespace holovibes
